import logging
from typing import TYPE_CHECKING, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models.booking import InterviewBooking
from app.services.embeddings import get_embedder

if TYPE_CHECKING:
    from app.db.vector_store import QdrantService
    from app.services.memory import RedisMemoryService

logger = logging.getLogger(__name__)

_RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You answer questions using only the retrieved passages below. Treat them as the single source of truth for facts related to documents, policies, and uploaded content.

Retrieved passages:
{combined_context}

Guidelines:
1. Every factual statement must be supported by the passages. If not present, clearly state that it is not available.
2. Do not assume or infer missing details.
3. If no relevant information is found, state that the answer cannot be determined from the available content.
4. If passages contain conflicting information, mention the conflict and summarize both sides.

Conversation history should only be used to understand context, not as a factual source.

If the user wants to book an interview:
- Collect name, email, date, and time.
- Only proceed once all details are available.
- After booking, respond normally without referring to tools or internal processes.
""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


def create_book_interview_tool(db: Session):
    @tool(description="Book an interview using name, email, date (YYYY-MM-DD), and time (HH:MM AM/PM).")
    def book_interview(
        name: str,
        email: str,
        date: str,
        time: str,
    ) -> str:
        """
        Creates a new interview booking in the database and returns a confirmation message.
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
                "Interview booked for %s on %s at %s (ID: %s)",
                name.strip(),
                date.strip(),
                time.strip(),
                new_booking.id,
            )

            return (
                f"Interview scheduled for {name.strip()} on {date.strip()} "
                f"at {time.strip()}. Booking ID: {new_booking.id}."
            )

        except Exception as e:
            logger.error("Booking failed: %s", e)
            db.rollback()
            return "Unable to complete the booking. Please check the details and try again."

    return book_interview


def _assistant_text(ai_msg) -> str:
    content = ai_msg.content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        text = " ".join(parts).strip()
        return text if text else str(content).strip()
    return str(content or "").strip()


class RAGService:
    def __init__(
        self,
        llm: BaseChatModel,
        qdrant: "QdrantService",
        memory: "RedisMemoryService",
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.memory = memory

    async def get_response(
        self,
        session_id: str,
        query: str,
        db: Session,
        document_id: Optional[str] = None,
    ) -> str:
        try:
            query_vector = get_embedder().embed_query(query)
            results = await self.qdrant.search_similar(
                query_vector, limit=15, document_id=document_id
            )

            context_blocks = []
            seen_texts = set()

            for res in results:
                text = res.get("text", "").strip()
                if not text or text in seen_texts:
                    continue

                seen_texts.add(text)

                doc_info = f"Document ID: {res.get('document_id', 'Unknown')}"
                if res.get("chunk_index") is not None:
                    doc_info += f" (Chunk {res.get('chunk_index')})"

                context_blocks.append(f"Source [{doc_info}]:\n{text}")

                if len(context_blocks) >= 5:
                    break

            combined_context = (
                "\n\n---\n\n".join(context_blocks)
                if context_blocks
                else "No relevant context found."
            )

        except Exception as e:
            logger.error("Context retrieval failed: %s", e)
            combined_context = "Unable to retrieve relevant documents."

        chat_history = await self.memory.get_chat_history(session_id)
        trimmed_history = chat_history[-10:] if len(chat_history) > 10 else chat_history

        prompt_value = await _RAG_CHAT_PROMPT.ainvoke(
            {
                "combined_context": combined_context,
                "history": trimmed_history,
                "input": query,
            }
        )

        messages = prompt_value.to_messages()
        user_msg = messages[-1]

        book_tool = create_book_interview_tool(db)
        llm_with_tools = self.llm.bind_tools([book_tool])

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)

            if getattr(ai_msg, "tool_calls", None):
                messages.append(ai_msg)

                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == book_tool.name:
                        try:
                            result = book_tool.invoke(tool_call["args"])
                        except Exception as e:
                            logger.error("Tool execution error: %s", e)
                            result = "Unable to complete the booking."

                        tool_msg = ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=result,
                        )
                        messages.append(tool_msg)

                ai_msg = await self.llm.ainvoke(messages)

            await self.memory.add_messages(session_id, [user_msg, ai_msg])

            return _assistant_text(ai_msg)

        except Exception:
            logger.exception("LLM response generation failed")
            return "The system is currently unavailable. Please try again later."