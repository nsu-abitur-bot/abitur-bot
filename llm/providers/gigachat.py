import logging
import os
from typing import List

from langchain_core.messages import BaseMessage
from langchain_gigachat.chat_models import GigaChat

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GigaChatProvider(BaseLLMProvider):
    def __init__(self):
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_PERS",
            model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            verify_ssl_certs=False,
        )
        logger.info(
            "GigaChat провайдер инициализирован (модель: %s)",
            os.getenv("GIGACHAT_MODEL", "GigaChat"),
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        response = await self.llm.ainvoke(messages)
        content = response.content.strip() if response.content else ""  # type: ignore
        return content
