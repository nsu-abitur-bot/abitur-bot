import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

load_dotenv()

# Загрузка конфигурации из переменных окружения
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет), "
    "отвечай только на вопросы связанные с университетом и поступлением. "
    "Отвечай коротко, долго не думай",
)

logger = logging.getLogger(__name__)

# Хранилище истории чатов в памяти
chat_store = {}


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Получить или создать историю чата для сессии"""
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
        logger.info(f"История чата создана для сессии: {session_id}")
    else:
        logger.info(f"История чата загружена для сессии: {session_id}")
    return chat_store[session_id]


# Инициализация LLM через Ollama
llm = ChatOllama(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    timeout=LLM_TIMEOUT,
)

# Создание промпт-шаблона
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Создание цепочки с историей
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)


def _clean_response(content: str) -> str:
    """Очистка ответа от служебных тегов"""
    # Удаляем <think>...</think>, если есть
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


async def ask_local_llm(message: str, session_id: str) -> str:
    """
    Отправка запроса к LLM с сохранением контекста диалога

    Args:
        message: сообщение от пользователя
        session_id: идентификатор переписки для сохранения контекста

    Returns:
        Ответ от LLM
    """
    try:
        logger.info(f"Обработка сообщения для сессии {session_id}")

        # Вызов цепочки с историей
        response = await chain_with_history.ainvoke(
            {"input": message}, config={"configurable": {"session_id": session_id}}
        )

        content = response.content
        logger.info(f"Ответ LLM получен для сессии {session_id}")

        # Очистка ответа
        content = _clean_response(content)

        if not content:
            logger.warning("Пустой ответ от LLM")
            return "Извините, я не смог сформулировать ответ"

        return content

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса к LLM: {e}", exc_info=True)
        return "Что-то пошло не так. Попробуйте позже."
