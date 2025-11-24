import logging
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from db.redis_client import RedisClient

load_dotenv()

logger = logging.getLogger(__name__)

LM_API_URL = "http://127.0.0.1:1234/v1"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_S.gguf"

SYSTEM_PROMPT = """
ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет),
отвечай только на вопросы связанные с университетом и поступлением.
Отвечай коротко, долго не думай.
"""

# Создаём клиент, совместимый с LM Studio API
llm = ChatOpenAI(
    base_url=LM_API_URL,
    model=MODEL,
    temperature=0.8,
)


async def ask_local_llm(message: str, session_id: str) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки,
    необходим для сохранения контекста (пока не используется)
    """
    redis_client = RedisClient()

    try:
        # Получаем историю переписки
        history = await redis_client.get_history(session_id)

        # Формируем список сообщений для LLM
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Добавляем историю переписки
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=content))  # type: ignore
            elif role == "assistant":
                messages.append(AIMessage(content=content))  # type: ignore

        # Добавляем текущее сообщение пользователя
        messages.append(HumanMessage(content=message))  # type: ignore

        response = await llm.ainvoke(messages)

        content = response.text().strip()

        # Удаляем блоки <think>...</think>
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        if not content:
            content = "Ответ не найден"

        # Сохраняем сообщение пользователя в историю
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        # Сохраняем ответ бота в историю
        await redis_client.add_message(
            session_id, {"role": "assistant", "content": content}
        )

        logger.debug(f"LLM response: {content}")
        return content

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Что-то пошло не так"
    finally:
        await redis_client.close()
