import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_rag_service
from app.db.session import get_db
from app.services.rag import RAGService
from app.schemas.chat import ChatRequest, ChatResponse
router = APIRouter(prefix="/chat", tags=["Conversational RAG"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
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
