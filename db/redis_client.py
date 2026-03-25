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
        logger.info(f"[Redis] Adding message to history for session: {session_id}")
        try:
            # pylance думает что эти функции могут быть синхронными, но они асинхронные
            # TODO: Исправить это и убрать type ignore
            await self.client.rpush(history_key, json.dumps(message))  # type: ignore
            await self.client.ltrim(history_key, -HISTORY_LIMIT, -1)  # type: ignore
            await self.client.expire(history_key, TTL_SECONDS)
            return await self.get_history(session_id)
        except redis.RedisError as e:
            logger.error(
                "[Redis] Ошибка добавления сообщения в Redis для %s: %s", session_id, e
            )
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

    async def clear_history(self, session_id: str) -> None:
        """Очищает историю чата для сессии."""
        history_key = f"chat_history:{session_id}"
        try:
            await self.client.delete(history_key)
            logger.debug("История очищена для сессии %s", session_id)
        except redis.RedisError as e:
            logger.error("Ошибка очистки истории для сессии %s: %s", session_id, e)
            raise

    async def set_awaiting_applicant_id(self, session_id: str, value: bool) -> None:
        """Устанавливает флаг ожидания идентификатора абитуриента."""
        key = f"session_state:{session_id}:awaiting_applicant_id"
        try:
            if value:
                await self.client.set(key, "1", ex=TTL_SECONDS)
            else:
                await self.client.delete(key)
            logger.debug(
                "Установлен awaiting_applicant_id=%s для %s", value, session_id
            )
        except redis.RedisError as e:
            logger.error("Ошибка установки awaiting_applicant_id: %s", e)
            raise

    async def is_awaiting_applicant_id(self, session_id: str) -> bool:
        """Проверяет, ожидается ли идентификатор абитуриента."""
        key = f"session_state:{session_id}:awaiting_applicant_id"
        try:
            val = await self.client.get(key)
            return val == "1"
        except redis.RedisError as e:
            logger.error("Ошибка проверки awaiting_applicant_id: %s", e)
            return False

    async def save_parsing_result(self, task_id: str, data: dict) -> None:
        """Сохраняет промежуточный результат парсинга в Redis."""
        key = f"parsing_task:{task_id}"
        try:
            await self.client.set(key, json.dumps(data), ex=TTL_SECONDS)
            logger.debug("Сохранен промежуточный результат парсинга для %s", task_id)
        except redis.RedisError as e:
            logger.error("Ошибка сохранения результата парсинга: %s", e)
            raise

    async def get_and_delete_parsing_result(self, task_id: str) -> dict | None:
        """Извлекает и удаляет промежуточный результат парсинга из Redis."""
        key = f"parsing_task:{task_id}"
        try:
            data = await self.client.get(key)
            if data:
                await self.client.delete(key)
                return json.loads(data)
            return None
        except redis.RedisError as e:
            logger.error("Ошибка извлечения результата парсинга: %s", e)
            return None

    # ------------------------------------------------------------------
    # Устаревшие псевдонимы (deprecated)
    # ------------------------------------------------------------------

    async def set_awaiting_snils(self, session_id: str, value: bool) -> None:  # noqa: E501
        """Deprecated: используйте set_awaiting_applicant_id."""
        await self.set_awaiting_applicant_id(session_id, value)

    async def is_awaiting_snils(self, session_id: str) -> bool:  # noqa: E501
        """Deprecated: используйте is_awaiting_applicant_id."""
        return await self.is_awaiting_applicant_id(session_id)

    async def close(self) -> None:
        """Закрывает соединение с Redis."""
        try:
            await self.client.aclose()
            logger.debug("Соединение с Redis закрыто")
        except redis.RedisError as e:
            logger.error("Ошибка закрытия соединения с Redis: %s", e)
