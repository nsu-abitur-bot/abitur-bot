"""Cloud.ru Foundation Models embeddings; no implicit retries or generation."""

import asyncio
import math
from typing import Any

import httpx

from config.instance import get_setting
from llm.base import BaseLLMProvider, LLMResult

MODEL = "ai-sage/Giga-Embeddings-instruct-3B"
BASE_URL = "https://foundation-models.api.cloud.ru/v1"


class CloudRuError(RuntimeError):
    """Safe error without upstream bodies, credentials or input text."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CloudRuEmbedding:
    embedding_dim = 2048

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model or get_setting("CLOUDRU_EMBEDDING_MODEL", MODEL)
        self._key = api_key or get_setting("CLOUDRU_API_KEY", "")
        self._transport = transport
        if not self._key:
            raise CloudRuError("Не задан API-ключ Cloud.ru")
        if self.model != MODEL:
            raise CloudRuError("Неизвестная модель эмбеддингов Cloud.ru")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise CloudRuError("Ожидаются непустые тексты для эмбеддингов")
        vectors: list[list[float]] = []
        try:
            with httpx.Client(
                transport=self._transport,
                timeout=60,
                follow_redirects=False,
            ) as client:
                for start in range(0, len(texts), 16):
                    batch = texts[start : start + 16]
                    response = client.post(
                        BASE_URL + "/embeddings",
                        headers={"Authorization": "Bearer " + self._key},
                        json={
                            "model": self.model,
                            "input": batch,
                            "encoding_format": "float",
                        },
                    )
                    if not response.is_success:
                        raise CloudRuError(
                            "Cloud.ru отклонил запрос эмбеддингов", response.status_code
                        )
                    body = response.json()
                    if body.get("model", self.model) != self.model:
                        raise CloudRuError("Cloud.ru вернул другую модель")
                    data = body["data"]
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise CloudRuError("Cloud.ru вернул неполный набор векторов")
                    ordered: dict[int, list[float]] = {}
                    for row in data:
                        index, vector = row["index"], row["embedding"]
                        if (
                            type(index) is not int
                            or not 0 <= index < len(batch)
                            or index in ordered
                            or not isinstance(vector, list)
                            or len(vector) != self.embedding_dim
                            or any(
                                type(x) not in (int, float) or not math.isfinite(x)
                                for x in vector
                            )
                            or not any(x != 0 for x in vector)
                        ):
                            raise CloudRuError("Cloud.ru вернул некорректные векторы")
                        ordered[index] = vector
                    vectors.extend(ordered[index] for index in range(len(batch)))
            return vectors
        except CloudRuError:
            raise
        except Exception:
            raise CloudRuError("Не удалось получить эмбеддинги Cloud.ru") from None

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aclose(self) -> None:
        # Each batch owns its client; callbacks can run in different event loops.
        pass


class CloudRuProvider(BaseLLMProvider):
    async def generate_with_usage(
        self, messages: Any, profile: Any = None, **kwargs: Any
    ) -> LLMResult:
        raise CloudRuError("Cloud.ru подключён только для эмбеддингов")

    def get_embeddings_model(self) -> CloudRuEmbedding:
        return CloudRuEmbedding()
