import json
import logging
from os import getenv
from typing import Dict, List

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
        self.client: redis_client.Redis = redis.from_url(
            REDIS_URL, decode_responses=True
        )

    async def add_message(
        self, session_id: str, message: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Добавляет сообщение в историю чата и возвращает историю."""
        history_key = f"chat_history:{session_id}"
        self.client.rpush
        try:
            # pylance думает что эти функции могут быть синхронными, но они асинхронные
            # TODO: Исправить это и убрать type ignore
            await self.client.rpush(history_key, json.dumps(message))  # type: ignore
            await self.client.ltrim(history_key, -HISTORY_LIMIT, -1)  # type: ignore
            await self.client.expire(history_key, TTL_SECONDS)
            logger.debug("Добавлено сообщение в историю %s", session_id)
            return await self.get_history(session_id)
        except redis.RedisError as e:
            logger.error("Ошибка добавления сообщения в Redis: %s", e)
            raise

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Получает историю чата."""
        history_key = f"chat_history:{session_id}"
        try:
            history_raw = await self.client.lrange(history_key, 0, -1)  # type: ignore
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

    async def close(self) -> None:
        """Закрывает соединение с Redis."""
        try:
            await self.client.aclose()
            logger.debug("Соединение с Redis закрыто")
        except redis.RedisError as e:
            logger.error("Ошибка закрытия соединения с Redis: %s", e)
