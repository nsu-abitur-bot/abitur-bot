import logging
import os
from typing import Any, List
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI

from llm.profiles import LLMProfiles

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


# Размерности для стандартных моделей эмбеддингов OpenAI
OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAILLM:
    """LLM-адаптер для LightRAG на базе OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        proxy_url: str | None = None,
    ) -> None:
        self.model = model
        self.profile = LLMProfiles.GRAPH
        self._async_http_client: httpx.AsyncClient | None = None
        effective_proxy = proxy_url or os.getenv("OPENAI_SOCKS5_PROXY")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(self.profile.timeout or 120),
        }
        if effective_proxy:
            try:
                self._async_http_client = httpx.AsyncClient(proxy=effective_proxy)
                client_kwargs["http_client"] = self._async_http_client
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось инициализировать SOCKS5 прокси "
                    "для OpenAI графового LLM. "
                    "Проверьте OPENAI_SOCKS5_PROXY (пример: socks5://user:pass@host:1080)."
                ) from exc

        self.client = AsyncOpenAI(**client_kwargs)
        logger.info(
            "OpenAILLM инициализирован (model=%s, proxy=%s)",
            self.model,
            _mask_proxy_url(effective_proxy),
        )

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            messages: Any = []
            if kwargs.get("system_prompt"):
                messages.append({"role": "system", "content": kwargs["system_prompt"]})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.responses.create(
                model=self.model,
                input=messages,
                max_output_tokens=int(
                    kwargs.get("max_tokens", self.profile.max_tokens or 2000)
                ),
                temperature=float(
                    kwargs.get("temperature", self.profile.temperature or 0.2)
                ),
            )

            return response.output_text.strip()
        except Exception as e:
            logger.error(f"Error in OpenAILLM: {e}")
            return ""


class OpenAIEmbedding:
    """Embedding-адаптер для LightRAG на базе OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        proxy_url: str | None = None,
    ) -> None:
        self.model = model
        self.profile = LLMProfiles.EMBEDDING
        self._async_http_client: httpx.AsyncClient | None = None
        effective_proxy = proxy_url or os.getenv("OPENAI_SOCKS5_PROXY")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(self.profile.timeout or 60),
        }
        if effective_proxy:
            try:
                self._async_http_client = httpx.AsyncClient(proxy=effective_proxy)
                client_kwargs["http_client"] = self._async_http_client
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось инициализировать SOCKS5 прокси для OpenAI эмбеддингов. "
                    "Проверьте OPENAI_SOCKS5_PROXY (пример: socks5://user:pass@host:1080)."
                ) from exc

        self.client = AsyncOpenAI(**client_kwargs)
        self.embedding_dim = OPENAI_EMBEDDING_DIMS.get(model, 1536)
        logger.info(
            "OpenAIEmbedding инициализирован (model=%s, proxy=%s)",
            self.model,
            _mask_proxy_url(effective_proxy),
        )

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error in OpenAIEmbedding: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]
