"""
Embeddings Service

Provides vector embedding strategies for chunks and queries:
- Local HuggingFace embeddings (via sentence-transformers)
- OpenAI embeddings
- Google Gemini embeddings
"""
from abc import ABC, abstractmethod
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

class BaseEmbedder(ABC):
    """Abstract Base Class for all Embedding providers."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text chunks."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string."""
        pass

class LocalHFEmbedder(BaseEmbedder):
    """Local HuggingFace model running via sentence-transformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

class OpenAIAPIEmbedder(BaseEmbedder):
    """Uses OpenAI's Embedding API."""
    def __init__(self, model_name: str = "text-embedding-3-small"):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing from environment variables.")
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

class GeminiAPIEmbedder(BaseEmbedder):
    """Uses Google's Gemini Embedding API."""
    def __init__(self, model_name: str = "models/embedding-001"):
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is missing.")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

def get_embedder(provider: str = "huggingface") -> BaseEmbedder:
    """
    Factory function to retrieve requested embedding provider.
    
    Args:
        provider: 'huggingface', 'openai', or 'gemini'
    """
    if provider == "huggingface":
        return LocalHFEmbedder()
    elif provider == "openai":
        return OpenAIAPIEmbedder()
    elif provider == "gemini":
        return GeminiAPIEmbedder()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
