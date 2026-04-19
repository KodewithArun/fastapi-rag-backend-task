from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _llm_yaml_path() -> Path:
    if settings.LLM_CONFIG_PATH:
        p = Path(settings.LLM_CONFIG_PATH)
        return p if p.is_absolute() else Path.cwd() / p
    return _project_root() / "config" / "llm.yaml"


class OpenaiLLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.3


class GeminiLLMConfig(BaseModel):
    model: str = "gemini-pro"
    temperature: float = 0.3


class GroqLLMConfig(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3


class LLMYamlConfig(BaseModel):
    provider: Literal["openai", "gemini", "groq"]
    openai: OpenaiLLMConfig = Field(default_factory=OpenaiLLMConfig)
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)
    groq: GroqLLMConfig = Field(default_factory=GroqLLMConfig)


@lru_cache(maxsize=1)
def _load_llm_yaml() -> LLMYamlConfig:
    path = _llm_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"LLM config not found at {path}. Copy config/llm.yaml or set LLM_CONFIG_PATH."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return LLMYamlConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_chat_llm() -> BaseChatModel:
    """Chat LLM from config/llm.yaml; keys from .env (cached)."""
    yaml_cfg = _load_llm_yaml()
    if yaml_cfg.provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env for the OpenAI provider."
            )
        return ChatOpenAI(
            model=yaml_cfg.openai.model,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=yaml_cfg.openai.temperature,
        )

    if yaml_cfg.provider == "groq":
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env for the Groq provider."
            )
        return ChatGroq(
            model=yaml_cfg.groq.model,
            api_key=settings.GROQ_API_KEY,
            temperature=yaml_cfg.groq.temperature,
        )

    api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is required for the Gemini provider."
        )
    return ChatGoogleGenerativeAI(
        model=yaml_cfg.gemini.model,
        google_api_key=api_key,
        temperature=yaml_cfg.gemini.temperature,
    )
