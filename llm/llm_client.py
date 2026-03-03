import logging
import re
from contextlib import suppress
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from db.redis_client import RedisClient
from llm.factory import get_llm_provider
from rag.retriever import query_graph

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """
ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет),
отвечай только на вопросы связанные с университетом и поступлением.
Отвечай коротко, долго не думай.

Используй следующую информацию из базы знаний для ответа на вопрос пользователя:

{context}

Если информации недостаточно, ответь по общим знаниям о НГУ
и честно укажи, что данных мало.
"""

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

        # 2. Получаем контекст из LightRAG
        try:
            rag_context = await query_graph(message)
            if not rag_context or rag_context.startswith("Error executing query"):
                rag_context = "Релевантный контекст из базы знаний не найден."
        except Exception as e:
            logger.warning(f"LightRAG query error: {e}")
            rag_context = "База знаний временно недоступна."

        # 3. Формируем сообщения и отправляем в LLM провайдер (Cerebras по умолчанию)
        try:
            system_prompt = SYSTEM_PROMPT_BASE.format(context=rag_context)
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Добавляем историю переписки
            history = await redis_client.get_history(session_id)
            for entry in history:
                role = entry.get("role", "")
                entry_content = entry.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=entry_content))
                elif role == "assistant":
                    messages.append(AIMessage(content=entry_content))

            provider = get_llm_provider()
            content = await provider.generate(messages)

            # Удаляем блоки <think>...</think>
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

            if not content:
                content = "Ответ не найден"
        except Exception as e:
            logger.warning(f"Provider generation error: {e}")
            content = "LLM временно недоступна."

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
