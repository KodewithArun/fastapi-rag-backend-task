from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Protocol

import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.core.config import settings


class Embedder(Protocol):
    """Embedder interface used by RAG/ingestion."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _embeddings_yaml_path() -> Path:
    if settings.EMBEDDING_CONFIG_PATH:
        p = Path(settings.EMBEDDING_CONFIG_PATH)
        return p if p.is_absolute() else Path.cwd() / p
    return _project_root() / "config" / "embeddings.yaml"


class HuggingFaceEmbeddingYaml(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class OpenaiEmbeddingYaml(BaseModel):
    model: str = "text-embedding-3-small"


class GeminiEmbeddingYaml(BaseModel):
    model: str = "models/embedding-001"


class EmbeddingsYamlConfig(BaseModel):
    provider: Literal["huggingface", "openai", "gemini"]
    huggingface: HuggingFaceEmbeddingYaml = Field(default_factory=HuggingFaceEmbeddingYaml)
    openai: OpenaiEmbeddingYaml = Field(default_factory=OpenaiEmbeddingYaml)
    gemini: GeminiEmbeddingYaml = Field(default_factory=GeminiEmbeddingYaml)


@lru_cache(maxsize=1)
def _load_embeddings_yaml() -> EmbeddingsYamlConfig:
    path = _embeddings_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding config not found at {path}. Copy config/embeddings.yaml or set EMBEDDING_CONFIG_PATH."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EmbeddingsYamlConfig.model_validate(raw)


class LocalHFEmbedder:
    def __init__(self, model_name: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class OpenAIAPIEmbedder:
    def __init__(self, model_name: str):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing from environment variables.")
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class GeminiAPIEmbedder:
    def __init__(self, model_name: str):
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is missing.")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Embedder from config/embeddings.yaml; keys from .env (cached)."""
    cfg = _load_embeddings_yaml()
    if cfg.provider == "huggingface":
        return LocalHFEmbedder(cfg.huggingface.model)
    if cfg.provider == "openai":
        return OpenAIAPIEmbedder(cfg.openai.model)
    return GeminiAPIEmbedder(cfg.gemini.model)
