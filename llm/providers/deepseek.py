import logging
import os
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import AsyncOpenAI

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения DEEPSEEK_API_KEY не установлена.")

        self.model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))
        self.timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=self.timeout_seconds,
        )
        logger.info(
            "DeepSeek провайдер инициализирован (модель: %s, timeout: %ss)",
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

    @staticmethod
    def _to_openai_message(
        message: BaseMessage,
    ) -> dict[str, str]:
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
