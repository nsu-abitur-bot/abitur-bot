from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage


class BaseLLMProvider(ABC):
    """Абстрактный интерфейс для LLM провайдера."""

    @abstractmethod
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Генерирует ответ по списку сообщений.

        Args:
            messages: Список сообщений (system, user, assistant)

        Returns:
            Текст ответа от модели
        """
