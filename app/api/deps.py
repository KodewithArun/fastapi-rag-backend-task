"""FastAPI deps: Qdrant, RAG, LLM (cached per process)."""

from functools import lru_cache

from app.db.vector_store import QdrantService
from app.services.llm_provider import get_chat_llm
from app.services.memory import get_memory_service
from app.services.rag import RAGService


@lru_cache(maxsize=1)
def get_qdrant_service() -> QdrantService:
    return QdrantService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService(
        llm=get_chat_llm(),
        qdrant=get_qdrant_service(),
        memory=get_memory_service(),
    )
