import logging
import re

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]

        response = await llm.ainvoke(messages)

        content = response.text().strip()

        # Удаляем блоки <think>...</think>
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        logger.debug(f"LLM response: {content}")

        return content or "Ответ не найден"

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Что-то пошло не так"
