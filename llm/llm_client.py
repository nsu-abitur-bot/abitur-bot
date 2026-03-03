import logging
from contextlib import suppress
from typing import Optional

from dotenv import load_dotenv

from db.redis_client import RedisClient
from rag.retriever import query_graph

load_dotenv()

logger = logging.getLogger(__name__)

# Создаем глобальный экземпляр для переиспользования соединения
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Получает или создает Redis клиент."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def ask_local_llm(message: str, session_id: str) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки,
    используется для получения и сохранения истории переписки (контекста)
    между пользователем и ассистентом
    """

    try:
        redis_client = await get_redis_client()

        # 1. Сначала сохраняем сообщение пользователя в историю
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        # 2. Используем LightRAG для получения ответа
        try:
            # LightRAG сам выполняет поиск и генерацию ответа
            content = await query_graph(message)

            logger.info(
                "\n=== ОТВЕТ LIGHTRAG ===\n%s\n================================\n",
                content,
            )

            if not content:
                content = "Ответ не найден в базе знаний."
        except Exception as e:
            logger.warning(f"Graph query error: {e}")
            content = "База знаний временно недоступна."

        # Сохраняем ответ бота в историю
        await redis_client.add_message(
            session_id, {"role": "assistant", "content": content}
        )

        return content

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Что-то пошло не так"


async def cleanup_redis():
    """Закрывает Redis соединение при завершении работы."""
    global _redis_client
    if _redis_client is not None:
        with suppress(Exception):
            await _redis_client.close()
        _redis_client = None
