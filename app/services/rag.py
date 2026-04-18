import logging
from typing import Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.db.vector_store import QdrantService
from app.models.booking import InterviewBooking
from app.services.embeddings import get_embedder
from app.services.llm_provider import get_llm_provider
from app.services.memory import get_memory_service

logger = logging.getLogger(__name__)


def create_book_interview_tool(db: Session):
    """Return a LangChain tool bound to the request-scoped DB session."""

    @tool
    def book_interview(
        name: str,
        email: str,
        date: str,
        time: str,
    ) -> str:
        """Book an interview for a candidate.

        Call only after the user has provided their full name, email, interview date (YYYY-MM-DD),
        and time (HH:MM AM/PM).
        """
        try:
            new_booking = InterviewBooking(
                name=name.strip(),
                email=email.strip(),
                date=date.strip(),
                time=time.strip(),
            )
            db.add(new_booking)
            db.commit()
            db.refresh(new_booking)

            logger.info(
                "Interview booked for %s on %s at %s (DB ID: %s)",
                name.strip(),
                date.strip(),
                time.strip(),
                new_booking.id,
            )
            return (
                f"Successfully booked interview for {name.strip()} on {date.strip()} "
                f"at {time.strip()}. Your booking ID is {new_booking.id}."
            )
        except Exception as e:
            logger.error("Failed to validate or process booking: %s", e)
            db.rollback()
            return (
                "Failed to book interview due to invalid information provided. Please try again."
            )

    return book_interview


class RAGService:
    def __init__(self, embed_provider: str = "huggingface"):
        self.llm_provider = get_llm_provider()
        self.embedder = get_embedder(embed_provider)
        self.qdrant = QdrantService()
        self.memory = get_memory_service()

    async def get_response(self, session_id: str, query: str, db: Session, document_id: Optional[str] = None) -> str:
        try:
            query_vector = self.embedder.embed_query(query)
            results = await self.qdrant.search_similar(query_vector, limit=8, document_id=document_id)

            context_blocks = []
            seen_texts = set()

            for res in results:
                score = res.get('score', 1.0)
                # Lowered score threshold for HF embeddings
                if score < 0.20:
                    continue

                text = res.get('text', '').strip()
                if not text or text in seen_texts:
                    continue

                seen_texts.add(text)
                doc_info = f"Document ID: {res.get('document_id', 'Unknown')}"
                if res.get('chunk_index') is not None:
                    doc_info += f" (Chunk {res.get('chunk_index')})"
                context_blocks.append(f"Source [{doc_info}]:\n{text}")

                if len(context_blocks) >= 5:
                    break

            combined_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            combined_context = "Could not retrieve documents at this time."

        chat_history = await self.memory.get_chat_history(session_id)
        trimmed_history = chat_history[-10:] if len(chat_history) > 10 else chat_history

        system_instruction = """You are a helpful AI assistant for Palm Mind AI.
You have access to the following retrieved documents to help answer user queries.
If the answer is not in the context, clearly state that you don't know based on the provided documents.

Context Information:
{combined_context}

You also have the ability to book interviews. 
If a user wants to book an interview, you MUST collect ALL of the following: Name, Email, Date, and Time.
Do not invoke the booking tool until you have gathered all 4 pieces of information."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction)
        ])

        sys_msg = prompt.format_messages(combined_context=combined_context)[0]

        messages = [sys_msg]
        messages.extend(trimmed_history)

        user_msg = HumanMessage(content=query)
        messages.append(user_msg)

        book_tool = create_book_interview_tool(db)
        base_llm = self.llm_provider.providers[0][1].llm
        try:
            llm_to_use = base_llm.bind_tools([book_tool])
        except Exception as e:
            logger.warning("Could not bind tools to LLM, fallback to normal generation: %s", e)
            llm_to_use = base_llm

        try:
            ai_msg = await llm_to_use.ainvoke(messages)

            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                messages.append(ai_msg)

                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == book_tool.name:
                        try:
                            booking_response = book_tool.invoke(tool_call["args"])
                        except Exception as e:
                            logger.error("book_interview tool invocation failed: %s", e)
                            booking_response = (
                                "Failed to book interview. Please try again."
                            )
                        tool_msg = ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=booking_response,
                        )
                        messages.append(tool_msg)

                ai_msg = await llm_to_use.ainvoke(messages)

            await self.memory.add_messages(session_id, [user_msg, ai_msg])

            content = ai_msg.content
            if isinstance(content, list):
                text_content = " ".join([c.get("text", "") for c in content if isinstance(c, dict) and "text" in c])
                if not text_content:
                    text_content = str(content)
                return text_content
            return str(content)

        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return "I'm sorry, I am currently experiencing issues."
