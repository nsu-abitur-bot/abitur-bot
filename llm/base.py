from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

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

    async def generate_stream(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Стримит ответ по кусочкам (дельтам).

        Дефолтная реализация — fallback на generate() с одной финальной дельтой,
        чтобы провайдеры без честной поддержки стриминга продолжали работать.
        """
        content = await self.generate(messages, profile=profile, **kwargs)
        if content:
            yield content

    def get_embeddings_model(self) -> Any:
        """Возвращает объект для работы с эмбеддингами.

        Объект должен поддерживать метод:
        embed_documents(texts: List[str]) -> List[List[float]].

        Если провайдер не поддерживает эмбеддинги, возвращает None.
        """
        return None
