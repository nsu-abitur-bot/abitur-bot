import logging
import os
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import AsyncOpenAI

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения OPENAI_API_KEY не установлена.")

        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.timeout_seconds,
        )
        logger.info(
            "OpenAI провайдер инициализирован (модель: %s, timeout: %ss)",
            self.model_name,
            self.timeout_seconds,
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        openai_messages: Any = [self._to_openai_message(m) for m in messages]

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content: Any = completion.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content or "")

    def get_embeddings_model(self) -> Any:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    @staticmethod
    def _to_openai_message(message: BaseMessage) -> dict[str, str]:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "user"

        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        return {"role": role, "content": content}
