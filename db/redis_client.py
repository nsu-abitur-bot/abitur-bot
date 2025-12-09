import json
import logging
from os import getenv
from typing import Any, Dict, List, cast

import redis.asyncio as redis
import redis.asyncio.client as redis_client
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = getenv("REDIS_URL")
HISTORY_LIMIT = 20  # Максимум сообщений в истории
TTL_SECONDS = 24 * 60 * 60  # 24 часа

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self.client: Any = redis.from_url(REDIS_URL, decode_responses=True)

    async def add_message(
        self, session_id: str, message: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Добавляет сообщение в историю чата и возвращает историю."""
        history_key = f"chat_history:{session_id}"
        try:
            # Cast bound methods to Any so Pyrefly doesn't treat their return
            # types as synchronous values (false positive).
            rpush_fn = cast(Any, self.client.rpush)
            ltrim_fn = cast(Any, self.client.ltrim)
            expire_fn = cast(Any, self.client.expire)

            await rpush_fn(history_key, json.dumps(message))
            await ltrim_fn(history_key, -HISTORY_LIMIT, -1)
            await expire_fn(history_key, TTL_SECONDS)
            logger.debug("Добавлено сообщение в историю %s", session_id)
            return await self.get_history(session_id)
        except redis.RedisError as e:
            logger.error("Ошибка добавления сообщения в Redis: %s", e)
            raise

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Получает историю чата."""
        history_key = f"chat_history:{session_id}"
        try:
            lrange_fn = cast(Any, self.client.lrange)
            history_raw = await lrange_fn(history_key, 0, -1)
            logger.debug(
                "Получена история для %s: %d сообщений", session_id, len(history_raw)
            )
            return [json.loads(msg) for msg in history_raw]
        except redis.RedisError as e:
            logger.error("Ошибка получения истории из Redis: %s", e)
            return []

    async def get(self, key: str) -> str | None:
        """Получает значение по ключу."""
        try:
            return await self.client.get(key)
        except redis.RedisError as e:
            logger.error("Ошибка получения ключа %s: %s", key, e)
            return None

    async def set(self, key: str, value: str) -> None:
        """Устанавливает значение по ключу."""
        try:
            await self.client.set(key, value)
            logger.debug("Установлен ключ %s", key)
        except redis.RedisError as e:
            logger.error("Ошибка установки ключа %s: %s", key, e)
            raise

    async def clear_history(self, session_id: str) -> None:
        """Очищает историю чата для сессии."""
        history_key = f"chat_history:{session_id}"
        try:
            await self.client.delete(history_key)
            logger.debug("История очищена для сессии %s", session_id)
        except redis.RedisError as e:
            logger.error("Ошибка очистки истории для сессии %s: %s", session_id, e)
            raise

    async def close(self) -> None:
        """Закрывает соединение с Redis."""
        try:
            await self.client.aclose()
            logger.debug("Соединение с Redis закрыто")
        except redis.RedisError as e:
            logger.error("Ошибка закрытия соединения с Redis: %s", e)
