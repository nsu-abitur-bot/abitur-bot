from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional

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


@dataclass
class ToolSpec:
    """Провайдер-независимое описание инструмента (function calling).

    Attributes:
        name: Имя функции, которое модель указывает при вызове.
        description: Описание для модели — когда и зачем вызывать инструмент.
        parameters: JSON Schema объекта аргументов (тип object).
    """

    name: str
    description: str
    parameters: dict


# Исполнитель инструмента: async (name, arguments) -> текст результата.
ToolExecutor = Callable[[str, dict], Awaitable[str]]


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

    async def generate_stream_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        """Стримит ответ и возвращает итоговый текст вместе с usage.

        Провайдеры, которые умеют отдавать usage в streaming API, переопределяют
        этот метод. Дефолт оставляет старое поведение без статистики токенов.
        """
        chunks: list[str] = []
        async for delta in self.generate_stream(messages, profile=profile, **kwargs):
            if not delta:
                continue
            chunks.append(delta)
            if on_delta is not None:
                await on_delta(delta)
        return LLMResult(text="".join(chunks).strip(), usage=LLMUsage())

    async def generate_with_tools(
        self,
        messages: List[BaseMessage],
        tools: List[ToolSpec],
        tool_executor: ToolExecutor,
        profile: Optional[LLMProfile] = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        """Генерация с поддержкой инструментов (function calling).

        Дефолтная реализация ИГНОРИРУЕТ инструменты и делегирует обычной
        генерации, чтобы провайдеры без нативной поддержки tool-calling
        продолжали работать. Провайдеры, умеющие вызывать функции,
        переопределяют этот метод.

        Если передан ``on_delta`` — финальный ответ стримится по дельтам, чтобы
        сохранить прогрессивное отображение в Telegram.
        """
        if on_delta is not None:
            return await self.generate_stream_with_usage(
                messages, profile=profile, on_delta=on_delta, **kwargs
            )
        return await self.generate_with_usage(messages, profile=profile, **kwargs)

    def get_embeddings_model(self) -> Any:
        """Возвращает объект для работы с эмбеддингами.

        Объект должен поддерживать метод:
        embed_documents(texts: List[str]) -> List[List[float]].

        Если провайдер не поддерживает эмбеддинги, возвращает None.
        """
        return None
