"""Redis chat history for RAG."""

import json
import logging
from typing import List

import redis.asyncio as redis
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.core.config import settings

logger = logging.getLogger(__name__)

MEMORY_FORMAT_VERSION = "1.0"


class RedisMemoryService:
    def __init__(self, prefix: str = "chat_history:", ttl_seconds: int = 86400, max_messages: int = 50):
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.max_messages = max_messages
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )

    def _serialize(self, messages: List[BaseMessage]) -> List[str]:
        msg_dicts = messages_to_dict(messages)
        return [
            json.dumps({"version": MEMORY_FORMAT_VERSION, "data": msg_dict}) for msg_dict in msg_dicts
        ]

    def _deserialize(self, raw_messages: List[str]) -> List[BaseMessage]:
        parsed_dicts = []
        for m in raw_messages:
            try:
                parsed = json.loads(m)
                if parsed.get("version") == MEMORY_FORMAT_VERSION:
                    parsed_dicts.append(parsed["data"])
                else:
                    parsed_dicts.append(parsed.get("data", parsed))
            except json.JSONDecodeError as e:
                logger.error("Malformed message JSON in Redis: %s", e)
        return messages_from_dict(parsed_dicts) if parsed_dicts else []

    async def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        key = f"{self.prefix}{session_id}"
        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.lrange(key, 0, -1)
            pipe.expire(key, self.ttl)
            results = await pipe.execute()
            raw_messages = results[0]

        if not raw_messages:
            return []
        return self._deserialize(raw_messages)

    async def add_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if not messages:
            return

        key = f"{self.prefix}{session_id}"
        serialized_msgs = self._serialize(messages)

        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *serialized_msgs)
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl)
            await pipe.execute()

    async def clear_history(self, session_id: str) -> None:
        await self.redis_client.delete(f"{self.prefix}{session_id}")

    async def close(self) -> None:
        await self.redis_client.aclose()


_MEMORY_SERVICE: RedisMemoryService | None = None


def get_memory_service() -> RedisMemoryService:
    global _MEMORY_SERVICE
    if _MEMORY_SERVICE is None:
        _MEMORY_SERVICE = RedisMemoryService()
    return _MEMORY_SERVICE
