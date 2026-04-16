"""
Chat Memory Service

Manages conversational history for multi-turn queries.
Implements an abstract base class with a Redis-backed storage engine
to seamlessly maintain the context of the user's RAG and booking chat.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List

import redis.asyncio as redis
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.core.config import settings

logger = logging.getLogger(__name__)

# Versioning for stored payload formats
MEMORY_FORMAT_VERSION = "1.0"

class BaseMemoryService(ABC):
    @abstractmethod
    async def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        pass

    @abstractmethod
    async def add_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        pass

    @abstractmethod
    async def clear_history(self, session_id: str) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class RedisMemoryService(BaseMemoryService):
    def __init__(self, prefix: str = "chat_history:", ttl_seconds: int = 86400, max_messages: int = 50):
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.max_messages = max_messages
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )

    def _serialize(self, messages: List[BaseMessage]) -> List[str]:
        """Defensive serialization with versioning."""
        try:
            msg_dicts = messages_to_dict(messages)
            return [json.dumps({"version": MEMORY_FORMAT_VERSION, "data": msg_dict}) for msg_dict in msg_dicts]
        except Exception as e:
            logger.error(f"Serialization error: {str(e)}")
            raise ValueError("Failed to serialize messages properly.") from e

    def _deserialize(self, raw_messages: List[str]) -> List[BaseMessage]:
        """Defensive deserialization handling versions."""
        parsed_dicts = []
        for m in raw_messages:
            try:
                parsed = json.loads(m)
                if parsed.get("version") == MEMORY_FORMAT_VERSION:
                    parsed_dicts.append(parsed["data"])
                else:
                    parsed_dicts.append(parsed.get("data", parsed))
            except json.JSONDecodeError as e:
                logger.error(f"Malformed message JSON in Redis: {e}")
                
        return messages_from_dict(parsed_dicts) if parsed_dicts else []

    async def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        start_time = time.time()
        key = f"{self.prefix}{session_id}"
        
        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.lrange(key, 0, -1)
                pipe.expire(key, self.ttl)
                results = await pipe.execute()
                raw_messages = results[0]
                
            if not raw_messages:
                return []
            
            messages = self._deserialize(raw_messages)
            logger.info(f"Fetched {len(messages)} messages for {session_id} in {time.time() - start_time:.4f}s")
            return messages
        except Exception as e:
            logger.error(f"Redis get_chat_history failed for {session_id}: {str(e)}")
            raise RuntimeError("Failed to fetch chat history") from e

    async def add_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if not messages:
            return
            
        start_time = time.time()
        key = f"{self.prefix}{session_id}"
        serialized_msgs = self._serialize(messages)
        
        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, *serialized_msgs)
                pipe.ltrim(key, -self.max_messages, -1)
                pipe.expire(key, self.ttl)
                await pipe.execute()
            
            logger.info(f"Added {len(messages)} messages for {session_id} in {time.time() - start_time:.4f}s")
        except Exception as e:
            logger.error(f"Redis add_messages failed for {session_id}: {str(e)}")
            raise RuntimeError("Failed to store chat history") from e

    async def clear_history(self, session_id: str) -> None:
        key = f"{self.prefix}{session_id}"
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Redis clear_history failed for {session_id}: {str(e)}")
            raise

    async def close(self) -> None:
        """Gracefully close the Redis connection pool."""
        await self.redis_client.aclose()


class InMemoryMemoryService(BaseMemoryService):
    """Fallback/Testing Provider."""
    def __init__(self):
        self.store = {}
        
    async def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        return self.store.get(session_id, [])
        
    async def add_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if session_id not in self.store:
            self.store[session_id] = []
        self.store[session_id].extend(messages)
        self.store[session_id] = self.store[session_id][-50:]
        
    async def clear_history(self, session_id: str) -> None:
        self.store.pop(session_id, None)
        
    async def close(self) -> None:
        pass


_MEMORY_SERVICE: BaseMemoryService = None

def get_memory_service(provider: str = "redis") -> BaseMemoryService:
    """Factory to retrieve the active Memory provider (caches the instance)."""
    global _MEMORY_SERVICE
    if _MEMORY_SERVICE is None:
        if provider == "redis":
            _MEMORY_SERVICE = RedisMemoryService()
        else:
            _MEMORY_SERVICE = InMemoryMemoryService()
    return _MEMORY_SERVICE
