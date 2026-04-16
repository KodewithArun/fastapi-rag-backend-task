"""
Custom RAG Service

Implements a completely custom Retrieval-Augmented Generation pipeline.
Strictly avoids LangChain's RetrievalQAChain.
Integrates Embeddings, Qdrant, Redis Memory, and the LLM.
Includes LLM tool calling for Interview Booking.
"""
import logging
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.services.llm_provider import get_llm_provider
from app.services.embeddings import get_embedder
from app.services.memory import get_memory_service
from app.db.vector_store import QdrantService
from app.core.config import settings

logger = logging.getLogger(__name__)

# Define the structure for Interview Booking
class BookInterview(BaseModel):
    """Call this tool to book an interview once you have collected the Name, Email, Date, and Time."""
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of the candidate")
    date: str = Field(description="Date of the interview (YYYY-MM-DD)")
    time: str = Field(description="Time of the interview (HH:MM AM/PM)")

class RAGService:
    def __init__(self, embed_provider: str = "openai"): # Defaulting to open ai for better RAG
        self.llm_provider = get_llm_provider()
        self.embedder = get_embedder(embed_provider)
        self.qdrant = QdrantService()
        self.memory = get_memory_service()
        
        # Attempt to bind tools to the LLM (OpenAI and Gemini both support this via LangChain)
        try:
            self.llm_with_tools = self.llm_provider.providers[0][1].llm.bind_tools([BookInterview])
        except Exception as e:
            logger.warning(f"Could not bind tools directly to LLM, fallback to normal generation: {e}")
            self.llm_with_tools = None

    async def _handle_booking(self, booking_args: dict, db) -> str:
        """
        Mock saving the booking info.
        In a full db setup, this would save to PostgreSQL using SQLAlchemy.
        """
        try:
            # Validate tool args with Pydantic model
            validated_booking = BookInterview(**booking_args)
            
            # Save to PostgreSQL
            from app.models.booking import InterviewBooking
            new_booking = InterviewBooking(
                name=validated_booking.name,
                email=validated_booking.email,
                date=validated_booking.date,
                time=validated_booking.time
            )
            db.add(new_booking)
            db.commit()
            db.refresh(new_booking)
            
            logger.info(f"*** INTERVIEW BOOKED: {validated_booking.model_dump_json()} (DB ID: {new_booking.id}) ***")
            return f"Successfully booked interview for {validated_booking.name} on {validated_booking.date} at {validated_booking.time}. Your booking ID is {new_booking.id}."
        except Exception as e:
            logger.error(f"Failed to validate or process booking arguments: {e}")
            if db:
                db.rollback()
            return "Failed to book interview due to invalid information provided. Please try again."

    async def get_response(self, session_id: str, query: str, db, document_id: Optional[str] = None) -> str:
        """
        The core of the custom RAG pipeline:
        1. Embed Query
        2. Retrieve Context from Qdrant
        3. Fetch Memory from Redis
        4. Build Prompt
        5. Invoke LLM and Handle Tools (with loop protection)
        6. Save Memory
        """
        # 1. Fetch relevant context with improved formatting (Source Separation)
        try:
            query_vector = self.embedder.embed_query(query)
            # Retrieve top 5 most similar chunks
            results = await self.qdrant.search_similar(query_vector, limit=5, document_id=document_id)
            
            context_blocks = []
            for res in results:
                doc_info = f"Document ID: {res.get('document_id', 'Unknown')} (Chunk {res.get('chunk_index', 'Unknown')})"
                context_blocks.append(f"Source [{doc_info}]:\n{res.get('text', '')}")
                
            combined_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            combined_context = "Could not retrieve documents at this time."

        # 2. Fetch Chat History & Trim for Context Limits
        chat_history = await self.memory.get_chat_history(session_id)
        # Token/context trimming: Keep max 10 recent messages (5 turns)
        trimmed_history = chat_history[-10:] if len(chat_history) > 10 else chat_history

        # 3. Construct System Prompt
        system_instruction = f"""You are a helpful AI assistant for Palm Mind AI.
You have access to the following retrieved documents to help answer user queries.
If the answer is not in the context, clearly state that you don't know based on the provided documents.

Context Information:
{combined_context}

You also have the ability to book interviews. 
If a user wants to book an interview, you MUST collect ALL of the following: Name, Email, Date, and Time.
Do not invoke the booking tool until you have gathered all 4 pieces of information.
"""
        
        # 4. Build message sequence
        messages = [SystemMessage(content=system_instruction)]
        messages.extend(trimmed_history)
        
        user_msg = HumanMessage(content=query)
        messages.append(user_msg)
        
        # Track ONLY the new messages for safe deduplicated appending to Memory layer
        new_messages_to_save = [user_msg]

        # 5. Invoke LLM with Loop Protection
        try:
            # We use the underlying LLM with tools bound
            llm_to_use = self.llm_with_tools if self.llm_with_tools else self.llm_provider.providers[0][1].llm
            
            max_iterations = 3
            current_iteration = 0
            
            while current_iteration < max_iterations:
                ai_msg = await llm_to_use.ainvoke(messages)
                
                messages.append(ai_msg)
                new_messages_to_save.append(ai_msg)
                
                # Break if no tool calls are requested
                if not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls:
                    break
                    
                # Handle Tool Calls (Interview Booking)
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "BookInterview":
                        # Process booking with Pydantic validation and DB integration
                        booking_response = await self._handle_booking(tool_call["args"], db)
                        
                        tool_msg = ToolMessage(tool_call_id=tool_call["id"], content=booking_response)
                        messages.append(tool_msg)
                        new_messages_to_save.append(tool_msg)
                        
                current_iteration += 1

            # 6. Save to Redis safely
            await self.memory.add_messages(session_id, new_messages_to_save)
            
            return str(ai_msg.content)
            
        except Exception as e:
            logger.error(f"LLM Generation failed during RAG: {e}")
            return "I'm sorry, I am currently experiencing issues connecting to my AI processor."

