import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique ID for the chat session, used for Redis Memory")
    query: str = Field(..., description="User's query")
    document_id: Optional[str] = Field(None, description="Optional Document ID to filter vectors by")

class ChatResponse(BaseModel):
    session_id: str
    reply: str

@router.post("/", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest, db: Session = Depends(get_db)):
    rag_service = RAGService()
    try:
        response_text = await rag_service.get_response(
            session_id=request.session_id,
            query=request.query,
            db=db,
            document_id=request.document_id
        )
        return ChatResponse(session_id=request.session_id, reply=response_text)
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error generating AI response.")
