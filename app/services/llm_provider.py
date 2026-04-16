"""
LLM Provider Service

Creates an Abstract Base Class for LLM interaction and implements an 
AutoSwitchingLLMProvider that attempts to use our primary LLM, 
and gracefully falls back to alternative providers (e.g. OpenAI -> Gemini)
if rate limits or outages occur.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract Base Class defining the contract for LLM generation."""
    @abstractmethod
    async def generate(self, messages: List[BaseMessage]) -> Any:
        """
        Generates a response using the configured LLM.
        
        Args:
            messages (List[BaseMessage]): A list of Langchain BaseMessages (System, Human, AI).
            
        Returns:
            Any: The raw generated AIMessage from the LLM.
        """
        pass

class OpenAIProvider(BaseLLMProvider):
    """Integration with OpenAI's Chat Models."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name, 
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )
        
    async def generate(self, messages: List[BaseMessage]) -> Any:
        return await self.llm.ainvoke(messages)

class GeminiProvider(BaseLLMProvider):
    """Integration with Google Gemini Chat Models."""
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=api_key,
            temperature=0.3
        )
        
    async def generate(self, messages: List[BaseMessage]) -> Any:
        return await self.llm.ainvoke(messages)

class AutoSwitchingLLMProvider(BaseLLMProvider):
    """
    Implements a resilient auto-switching feature.
    If the first mapped API provider fails (e.g. due to rate limits or API outage),
    it immediately falls back to the next available provider.
    """
    def __init__(self):
        self.providers = []
        
        # Priority 1: OpenAI
        if settings.OPENAI_API_KEY and "your_" not in settings.OPENAI_API_KEY:
            self.providers.append(("OpenAI", OpenAIProvider()))
            
        # Priority 2: Gemini (Fallback)
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        if api_key and "your_" not in api_key:
            self.providers.append(("Gemini", GeminiProvider()))
            
        if not self.providers:
            # We don't raise an error strictly on init to prevent app crash if keys are missing initially,
            # but we will log a severe warning.
            logger.warning("No valid LLM providers configured in .env (OpenAI or Gemini).")

    async def generate(self, messages: List[BaseMessage]) -> Any:
        if not self.providers:
            raise RuntimeError("No LLM API keys configured. Please add OPENAI_API_KEY or GEMINI_API_KEY.")

        last_exception = None
        for name, provider in self.providers:
            try:
                # logger.debug(f"Attempting generation with {name}...")
                response = await provider.generate(messages)
                return response
            except Exception as e:
                logger.warning(f"LLM Provider '{name}' failed: {str(e)}. Switching to next provider...")
                last_exception = e
                
        logger.error("All auto-switching LLM providers failed.")
        raise RuntimeError(f"LLM Generation failed across all providers. Last error: {str(last_exception)}")

def get_llm_provider() -> BaseLLMProvider:
    """Factory to retrieve our resilient auto-switching LLM provider."""
    return AutoSwitchingLLMProvider()
