import logging
import re
from contextlib import suppress
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from db.redis_client import RedisClient
from llm.factory import get_llm_provider
from rag.retriever import search_similar

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """
ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет),
отвечай только на вопросы связанные с университетом и поступлением.
Отвечай коротко, долго не думай.

Используй следующую информацию из базы знаний для ответа на вопрос пользователя:

{context}

Если информация в базе знаний не помогает ответить на вопрос, отвечай на основе общих знаний о НГУ.
"""  # noqa: E501

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
        history = await redis_client.add_message(
            session_id, {"role": "user", "content": message}
        )

        # 2. Ищем релевантный контекст в векторной базе данных
        try:
            similar_docs = search_similar(message, k=3)
            context = "\n\n".join([doc.page_content for doc in similar_docs])

            logger.info(
                f"\n=== НАЙДЕННЫЙ КОНТЕКСТ (RAG) ===\n{context}\n================================\n"  # noqa: E501
            )

            if not context:
                context = "Релевантной информации не найдено в базе знаний."
        except Exception as e:
            logger.warning(f"RAG search error: {e}")
            context = "База знаний временно недоступна."

        # Формируем список сообщений для LLM
        system_prompt = SYSTEM_PROMPT_BASE.format(context=context)
        messages = [SystemMessage(content=system_prompt)]

        # Добавляем историю переписки
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=content))  # type: ignore
            elif role == "assistant":
                messages.append(AIMessage(content=content))  # type: ignore

        # Отправляем запрос в LLM через провайдер
        provider = get_llm_provider()
        content = await provider.generate(messages)  # type: ignore

        # Удаляем блоки <think>...</think>
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        if not content:
            content = "Ответ не найден"

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
