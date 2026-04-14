import logging
import os
from typing import Any, List
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import AsyncOpenAI

from llm.base import BaseLLMProvider

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
        # self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
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

    async def generate(self, messages: List[BaseMessage]) -> str:
        openai_messages: Any = [self._to_openai_message(m) for m in messages]

        try:
            completion = await self.client.responses.create(
                model=self.model_name,
                input=openai_messages,
                # max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception:
            logger.exception(
                "Ошибка запроса к OpenAI (proxy=%s)",
                _mask_proxy_url(self.proxy_url),
            )
            raise

        content: Any = completion.output_text
        return content.strip() if isinstance(content, str) else str(content or "")

    def get_embeddings_model(self) -> Any:
        from langchain_openai import OpenAIEmbeddings

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
