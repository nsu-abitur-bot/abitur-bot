from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage

from llm.profiles import LLMProfile


@dataclass
class LLMUsage:
    """Сведения о потраченных токенах за одну генерацию."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResult:
    """Результат генерации LLM: текст + информация о токенах."""

    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)


class BaseLLMProvider(ABC):
    """Абстрактный интерфейс для LLM провайдера."""

    @abstractmethod
    async def generate_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> LLMResult:
        """Генерирует ответ + возвращает информацию о потраченных токенах."""
        pass

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
        result = await self.generate_with_usage(messages, profile=profile, **kwargs)
        return result.text

    def get_embeddings_model(self) -> Any:
        """Возвращает объект для работы с эмбеддингами.

        Объект должен поддерживать метод:
        embed_documents(texts: List[str]) -> List[List[float]].

        Если провайдер не поддерживает эмбеддинги, возвращает None.
        """
        return None
