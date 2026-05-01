from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage

from llm.profiles import LLMProfile


class BaseLLMProvider(ABC):
    """Абстрактный интерфейс для LLM провайдера."""

    @abstractmethod
    async def generate(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> str:
        """Генерирует ответ по списку сообщений.

        Args:
            messages: Список сообщений (system, user, assistant)
            profile: Объект с настройками лимитов (max_tokens, timeout, temperature)
            kwargs: Прочие параметры генерации

        Returns:
            Текст ответа от модели
        """
        pass

    def get_embeddings_model(self) -> Any:
        """Возвращает объект для работы с эмбеддингами.

        Объект должен поддерживать метод:
        embed_documents(texts: List[str]) -> List[List[float]].

        Если провайдер не поддерживает эмбеддинги, возвращает None.
        """
        return None
