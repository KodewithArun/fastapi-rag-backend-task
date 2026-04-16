import logging
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: List[BaseMessage]) -> Any:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model_name: str | None = None):
        self.llm = ChatOpenAI(
            model=model_name or settings.OPENAI_MODEL_NAME, 
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )
        
    async def generate(self, messages: List[BaseMessage]) -> Any:
        return await self.llm.ainvoke(messages)

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str | None = None):
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.llm = ChatGoogleGenerativeAI(
            model=model_name or settings.GEMINI_MODEL_NAME, 
            google_api_key=api_key,
            temperature=0.3
        )
        
    async def generate(self, messages: List[BaseMessage]) -> Any:
        return await self.llm.ainvoke(messages)

class AutoSwitchingLLMProvider(BaseLLMProvider):
    def __init__(self):
        self.providers = []
        
        if settings.OPENAI_API_KEY and "your_" not in settings.OPENAI_API_KEY:
            self.providers.append(("OpenAI", OpenAIProvider()))
            
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        if api_key and "your_" not in api_key:
            self.providers.append(("Gemini", GeminiProvider()))
            
        if not self.providers:
            logger.warning("No valid LLM providers configured.")

    async def generate(self, messages: List[BaseMessage]) -> Any:
        if not self.providers:
            raise RuntimeError("No LLM API keys configured. Please add OPENAI_API_KEY or GEMINI_API_KEY.")

        last_exception = None
        for name, provider in self.providers:
            try:
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
