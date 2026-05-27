import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.base import BaseLLMProvider, LLMResult, LLMUsage
from llm.profiles import LLMProfile

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения GEMINI_API_KEY не установлена.")

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

        self.client = genai.Client(api_key=api_key)

        logger.info(
            "Gemini провайдер инициализирован (модель: %s)",
            self.model_name,
        )

    def _build_request(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile],
    ) -> tuple[list[dict[str, Any]], types.GenerateContentConfig]:
        system_instruction = None
        gemini_messages: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = str(msg.content)
            elif isinstance(msg, AIMessage):
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": str(msg.content)}]}
                )
            elif isinstance(msg, HumanMessage):
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": str(msg.content)}]}
                )
            else:
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": str(msg.content)}]}
                )

        temperature = (
            profile.temperature
            if profile and profile.temperature is not None
            else self.temperature
        )
        max_tokens = profile.max_tokens if profile else None

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        return gemini_messages, config

    async def generate_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> LLMResult:
        gemini_messages, config = self._build_request(messages, profile)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=gemini_messages,
            config=config,
        )

        content = response.text
        text = content.strip() if isinstance(content, str) else str(content or "")

        usage = LLMUsage()
        raw_usage = getattr(response, "usage_metadata", None)
        if raw_usage is not None:
            usage = _usage_from_gemini_metadata(raw_usage)

        return LLMResult(text=text, usage=usage)

    async def generate_stream(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        gemini_messages, config = self._build_request(messages, profile)

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=gemini_messages,
            config=config,
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def generate_stream_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        gemini_messages, config = self._build_request(messages, profile)

        chunks: list[str] = []
        usage = LLMUsage()
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=gemini_messages,
            config=config,
        )
        async for chunk in stream:
            raw_usage = getattr(chunk, "usage_metadata", None)
            if raw_usage is not None:
                usage = _usage_from_gemini_metadata(raw_usage)

            text = getattr(chunk, "text", None)
            if not text:
                continue
            chunks.append(text)
            if on_delta is not None:
                await on_delta(text)

        return LLMResult(text="".join(chunks).strip(), usage=usage)

    def get_embeddings_model(self) -> Any:
        return GeminiEmbeddings(self.client)


class GeminiEmbeddings:
    """Воркер для эмбеддингов Gemini, совместимый с FAQMatcher."""

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # google-genai SDK поддерживает пакетную обработку
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        # response.embeddings — список объектов с полем values
        if not response.embeddings:
            return []
        return [list(e.values or []) for e in response.embeddings]


def _usage_from_gemini_metadata(raw_usage: Any) -> LLMUsage:
    prompt_tokens = int(getattr(raw_usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(raw_usage, "candidates_token_count", 0) or 0)
    total_tokens = int(
        getattr(raw_usage, "total_token_count", 0)
        or (prompt_tokens + completion_tokens)
    )
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
