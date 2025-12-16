import logging
import os
import re
from contextlib import suppress
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from db.redis_client import RedisClient
from rag.retriever import search_similar

load_dotenv()

logger = logging.getLogger(__name__)

LM_API_URL = "http://127.0.0.1:1234/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_S.gguf"

SYSTEM_PROMPT_BASE = """
ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет),
отвечай только на вопросы связанные с университетом и поступлением.
Отвечай коротко, долго не думай.

Используй следующую информацию из базы знаний для ответа на вопрос пользователя:

{context}

Если информация в базе знаний не помогает ответить на вопрос, отвечай на основе общих знаний о НГУ.
"""  # noqa: E501

# Создаём клиент, совместимый с GigaChat API
llm = GigaChat(
    credentials=os.getenv("OPENAI_API_KEY"),
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    verify_ssl_certs=False,
)

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
                # type: ignore нужен потому что mypy не может определить,
                # что HumanMessage.content принимает str, хотя это валидно
                messages.append(HumanMessage(content=content))  # type: ignore
            elif role == "assistant":
                # Аналогично для AIMessage.content
                messages.append(AIMessage(content=content))  # type: ignore

        # Отправляем запрос в LLM
        response = await llm.ainvoke(messages)

        # type: ignore нужен потому что mypy не может определить,
        # что content принимает str, хотя это валидно
        content = response.content.strip() if response.content else ""  # type: ignore

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
        # Используем contextlib.suppress для безопасного закрытия соединения
        # Это предотвращает маскирование оригинальной ошибки, если close() упадет
        with suppress(Exception):
            await _redis_client.close()
        _redis_client = None
