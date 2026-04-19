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
            """
You answer questions using ONLY the retrieved passages.

Rules:
1. Do NOT hallucinate or assume missing facts.
2. If information is missing, clearly say so.
3. If no relevant context exists, say it cannot be determined.

IMPORTANT TOOL RULES:
- If a tool is used, DO NOT modify or reinterpret tool output.
- Tool output is FINAL and must be returned as-is.
- Do NOT add or remove fields from tool results.
""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)



def create_book_interview_tool(db: Session):
    @tool(description="Book an interview using name, email, date (YYYY-MM-DD), and time (HH:MM AM/PM).")
    def book_interview(name: str, email: str, date: str, time: str):
        """
        Creates a new interview booking in DB and returns structured data.
        Prevents duplicate bookings for the same email, date, and time.
        """
        try:
            # Check for duplicate booking
            existing = db.query(InterviewBooking).filter(
                InterviewBooking.email == email.strip(),
                InterviewBooking.date == date.strip(),
                InterviewBooking.time == time.strip()
            ).first()
            if existing:
                logger.info(
                    "Duplicate booking attempt: %s | %s | %s | ID=%s",
                    name, date, time, existing.id
                )
                return {
                    "status": "error",
                    "message": "Duplicate booking: An interview is already scheduled for this email, date, and time.",
                    "booking_id": existing.id,
                    "name": existing.name,
                    "email": existing.email,
                    "date": existing.date,
                    "time": existing.time
                }

            booking = InterviewBooking(
                name=name.strip(),
                email=email.strip(),
                date=date.strip(),
                time=time.strip(),
            )

            db.add(booking)
            db.commit()
            db.refresh(booking)

            logger.info(
                "Interview booked: %s | %s | %s | ID=%s",
                name, date, time, booking.id
            )

            return {
                "status": "success",
                "name": name.strip(),
                "email": email.strip(),
                "date": date.strip(),
                "time": time.strip(),
                "booking_id": booking.id,
                "message": "Interview successfully scheduled."
            }

        except Exception as e:
            logger.error("Booking failed: %s", e)
            db.rollback()
            return {
                "status": "error",
                "message": "Unable to complete booking. Please try again."
            }

    return book_interview


def _assistant_text(ai_msg) -> str:
    content = ai_msg.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        ).strip()
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
            seen = set()

            for r in results:
                text = r.get("text", "").strip()
                if not text or text in seen:
                    continue

                seen.add(text)

                doc_info = f"Document ID: {r.get('document_id', 'Unknown')}"
                if r.get("chunk_index") is not None:
                    doc_info += f" (Chunk {r['chunk_index']})"

                context_blocks.append(f"[{doc_info}]\n{text}")

                if len(context_blocks) >= 5:
                    break

            combined_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            combined_context = "Unable to retrieve context."

        chat_history = await self.memory.get_chat_history(session_id)
        trimmed_history = chat_history[-10:]

        prompt_value = await _RAG_CHAT_PROMPT.ainvoke(
            {
                "combined_context": combined_context,
                "history": trimmed_history,
                "input": query,
            }
        )

        messages = prompt_value.to_messages()

        book_tool = create_book_interview_tool(db)
        llm_with_tools = self.llm.bind_tools([book_tool])

        try:
            ai_msg = await llm_with_tools.ainvoke(messages)

            if getattr(ai_msg, "tool_calls", None):
                messages.append(ai_msg)

                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == book_tool.name:

                        result = book_tool.invoke(tool_call["args"])

                        tool_msg = ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=str(result),
                        )

                        messages.append(tool_msg)

                        await self.memory.add_messages(session_id, [tool_msg])

                        return str(result)

            await self.memory.add_messages(session_id, [ai_msg])

            return _assistant_text(ai_msg)

        except Exception as e:
            logger.exception("LLM failure: %s", e)
            return "System error. Please try again later."