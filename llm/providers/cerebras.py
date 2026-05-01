import asyncio
import logging
import os
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.base import BaseLLMProvider
from llm.profiles import LLMProfile

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

        self.model_name = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        self.max_completion_tokens = int(
            os.getenv("CEREBRAS_MAX_COMPLETION_TOKENS", "1024")
        )
        self.temperature = float(os.getenv("CEREBRAS_TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("CEREBRAS_TOP_P", "1"))
        self.timeout_seconds = float(os.getenv("CEREBRAS_TIMEOUT_SECONDS", "60"))
        self.max_retries = int(os.getenv("CEREBRAS_MAX_RETRIES", "2"))

        Cerebras = _load_cerebras_class()
        self.client = Cerebras(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        logger.info(
            "Cerebras провайдер инициализирован (модель: %s, "
            "timeout: %ss, retries: %s)",
            self.model_name,
            self.timeout_seconds,
            self.max_retries,
        )

    async def generate(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> str:
        cerebras_messages: Any = [
            self._to_cerebras_message(message) for message in messages
        ]

        temperature = profile.temperature if profile and profile.temperature is not None else self.temperature
        max_completion_tokens = profile.max_tokens if profile else self.max_completion_tokens
        timeout = profile.timeout if profile and profile.timeout is not None else self.timeout_seconds

        completion: Any = await asyncio.to_thread(
            self.client.chat.completions.create,
            messages=cerebras_messages,
            model=self.model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=self.top_p,
            stream=False,
            timeout=timeout,
        )

        content: Any = completion.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content or "")

    def get_embeddings_model(self) -> Any:
        return None

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
