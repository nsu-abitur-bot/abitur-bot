import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from llm.base import BaseLLMProvider, LLMResult, LLMUsage
from llm.profiles import LLMProfile

logger = logging.getLogger(__name__)


def _mask_proxy_url(proxy_url: str | None) -> str:
    if not proxy_url:
        return "disabled"
    try:
        parsed = urlsplit(proxy_url)
        host = parsed.hostname or "unknown-host"
        port = parsed.port or "unknown-port"
        username = parsed.username
        credentials = f"{username}:***@" if username else ""
        return f"{parsed.scheme}://{credentials}{host}:{port}"
    except Exception:
        return "invalid-proxy-url"


class OpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения OPENAI_API_KEY не установлена.")

        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        self.proxy_url = os.getenv("OPENAI_SOCKS5_PROXY")

        self._async_http_client: httpx.AsyncClient | None = None
        self._sync_http_client: httpx.Client | None = None
        if self.proxy_url:
            try:
                self._async_http_client = httpx.AsyncClient(proxy=self.proxy_url)
                self._sync_http_client = httpx.Client(proxy=self.proxy_url)
            except ImportError as exc:
                raise RuntimeError(
                    "OPENAI_SOCKS5_PROXY задан, но не установлена "
                    "SOCKS-поддержка для httpx. "
                    'Установите зависимость: pip install "httpx[socks]"'
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось инициализировать SOCKS5 прокси для OpenAI. "
                    "Проверьте OPENAI_SOCKS5_PROXY (пример: socks5://user:pass@host:1080)."
                ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout_seconds,
        }
        if self._async_http_client is not None:
            client_kwargs["http_client"] = self._async_http_client

        self.client = AsyncOpenAI(
            **client_kwargs,
        )
        logger.info(
            "OpenAI провайдер инициализирован (модель: %s, timeout: %ss, proxy: %s)",
            self.model_name,
            self.timeout_seconds,
            _mask_proxy_url(self.proxy_url),
        )

    def _resolve_params(
        self, profile: Optional[LLMProfile]
    ) -> tuple[float, int, float]:
        temperature = (
            profile.temperature
            if profile and profile.temperature is not None
            else self.temperature
        )
        max_tokens = (
            profile.max_tokens
            if profile and profile.max_tokens is not None
            else self.max_tokens
        )
        timeout = (
            profile.timeout
            if profile and profile.timeout is not None
            else self.timeout_seconds
        )
        return temperature, max_tokens, timeout

    async def generate_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> LLMResult:
        openai_messages: Any = [self._to_openai_message(m) for m in messages]
        temperature, max_tokens, timeout = self._resolve_params(profile)

        try:
            completion = await self.client.responses.create(
                model=self.model_name,
                input=openai_messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception:
            logger.exception(
                "Ошибка запроса к OpenAI (proxy=%s)",
                _mask_proxy_url(self.proxy_url),
            )
            raise

        content: Any = completion.output_text
        text = content.strip() if isinstance(content, str) else str(content or "")

        usage = LLMUsage()
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            usage = _usage_from_openai_metadata(raw_usage)

        return LLMResult(text=text, usage=usage)

    async def generate_stream(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        openai_messages: Any = [self._to_openai_message(m) for m in messages]
        temperature, max_tokens, timeout = self._resolve_params(profile)

        try:
            async with self.client.responses.stream(
                model=self.model_name,
                input=openai_messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            ) as stream:
                async for event in stream:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield delta
        except Exception:
            logger.exception(
                "Ошибка стриминга OpenAI (proxy=%s)",
                _mask_proxy_url(self.proxy_url),
            )
            raise

    async def generate_stream_with_usage(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        openai_messages: Any = [self._to_openai_message(m) for m in messages]
        temperature, max_tokens, timeout = self._resolve_params(profile)

        chunks: list[str] = []
        usage = LLMUsage()
        try:
            async with self.client.responses.stream(
                model=self.model_name,
                input=openai_messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            ) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if not delta:
                            continue
                        chunks.append(delta)
                        if on_delta is not None:
                            await on_delta(delta)
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        raw_usage = getattr(response, "usage", None)
                        if raw_usage is not None:
                            usage = _usage_from_openai_metadata(raw_usage)
        except Exception:
            logger.exception(
                "Ошибка стриминга OpenAI (proxy=%s)",
                _mask_proxy_url(self.proxy_url),
            )
            raise

        return LLMResult(text="".join(chunks).strip(), usage=usage)

    def get_embeddings_model(self) -> Any:
        embedding_kwargs: dict[str, Any] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        }
        if self._sync_http_client is not None:
            embedding_kwargs["http_client"] = self._sync_http_client

        return OpenAIEmbeddings(
            **embedding_kwargs,
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


def _usage_from_openai_metadata(raw_usage: Any) -> LLMUsage:
    input_tokens = (
        getattr(raw_usage, "input_tokens", None)
        or getattr(raw_usage, "prompt_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(raw_usage, "output_tokens", None)
        or getattr(raw_usage, "completion_tokens", None)
        or 0
    )
    total_tokens = (
        getattr(raw_usage, "total_tokens", None)
        or (int(input_tokens) + int(output_tokens))
    )
    return LLMUsage(
        prompt_tokens=int(input_tokens),
        completion_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
    )
