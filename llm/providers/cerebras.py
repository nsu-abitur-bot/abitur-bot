import asyncio
import logging
import os
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _load_cerebras_class():
    try:
        from cerebras.cloud.sdk import Cerebras

        return Cerebras
    except ImportError as exc:
        raise RuntimeError(
            "Пакет cerebras-cloud-sdk не установлен. "
            "Установите его: pip install cerebras-cloud-sdk"
        ) from exc


class CerebrasProvider(BaseLLMProvider):
    def __init__(self):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Переменная окружения CEREBRAS_API_KEY не установлена. "
                "Пожалуйста, настройте её для использования провайдера Cerebras."
            )

        self.model_name = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
        self.max_completion_tokens = int(
            os.getenv("CEREBRAS_MAX_COMPLETION_TOKENS", "1024")
        )
        self.temperature = float(os.getenv("CEREBRAS_TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("CEREBRAS_TOP_P", "1"))

        Cerebras = _load_cerebras_class()
        self.client = Cerebras(api_key=api_key)
        logger.info(
            "Cerebras провайдер инициализирован (модель: %s)",
            self.model_name,
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        cerebras_messages = [self._to_cerebras_message(message) for message in messages]

        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            messages=cerebras_messages,
            model=self.model_name,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
        )

        content: Any = completion.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content or "")

    @staticmethod
    def _to_cerebras_message(message: BaseMessage) -> dict[str, str]:
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
