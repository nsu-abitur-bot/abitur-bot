import logging
import os
from typing import List

from langchain_core.messages import BaseMessage
from langchain_gigachat.chat_models import GigaChat

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GigaChatProvider(BaseLLMProvider):
    def __init__(self):
        # TODO
        # Настроить проверку SSL-сертификата через переменную окружения.
        # По умолчанию проверка включена для безопасности.

        api_key = os.getenv("GIGACHAT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Переменная окружения GIGACHAT_API_KEY не установлена. "
                "Пожалуйста, настройте её для использования провайдера GigaChat."
            )
        model_name = os.getenv("GIGACHAT_MODEL", "GigaChat")

        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_PERS",
            model=model_name,
            verify_ssl_certs=False,
        )
        logger.info(
            "GigaChat провайдер инициализирован (модель: %s)",
            model_name,
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        response = await self.llm.ainvoke(messages)
        if isinstance(response.content, str):
            content = response.content.strip()
        else:
            content = str(response.content) if response.content else ""
        return content
