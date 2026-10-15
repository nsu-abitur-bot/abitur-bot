"""Сбой эмбеддинга — ошибка, а не нулевой вектор (#283).

Нулевой вектор в nano_vectordb даёт NaN-скоры: на запросе — случайные top-k
чанки, на записи — NaN-строку, которая до переиндексации всплывает в топ-1
любого запроса. Поэтому адаптер обязан падать, повторы делает LightRAG.
"""

from types import SimpleNamespace

import pytest

from llm.providers.gemini_graph_adapters import GeminiEmbedding
from llm.providers.openai_graph_adapters import OpenAIEmbedding


class _Failing:
    async def __call__(self, *args, **kwargs):
        raise RuntimeError("RateLimitError 429")


def _openai_embedding(create) -> OpenAIEmbedding:
    adapter = OpenAIEmbedding(api_key="test-key")
    adapter.client = SimpleNamespace(  # type: ignore[assignment]
        embeddings=SimpleNamespace(create=create)
    )
    return adapter


def _gemini_embedding(embed_content) -> GeminiEmbedding:
    adapter = GeminiEmbedding(api_key="test-key")
    adapter.client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    )
    return adapter


async def test_openai_embedding_raises_instead_of_zero_vector():
    adapter = _openai_embedding(_Failing())

    with pytest.raises(RuntimeError, match="429"):
        await adapter(["Когда приём документов?"])


async def test_openai_embedding_returns_vectors_on_success():
    async def create(model, input):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input]
        )

    adapter = _openai_embedding(create)

    assert await adapter(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


async def test_gemini_embedding_raises_instead_of_zero_vector():
    adapter = _gemini_embedding(_Failing())

    with pytest.raises(RuntimeError, match="429"):
        await adapter(["Когда приём документов?"])


@pytest.mark.parametrize(
    "embeddings",
    [
        None,
        [],
        [SimpleNamespace(values=[0.1, 0.2])],
        [SimpleNamespace(values=[0.1, 0.2]), SimpleNamespace(values=None)],
    ],
)
async def test_gemini_embedding_raises_on_missing_vectors(embeddings):
    """Неполный ответ модели раньше тоже добивался нулями."""

    async def embed_content(model, contents, config):
        return SimpleNamespace(embeddings=embeddings)

    adapter = _gemini_embedding(embed_content)

    with pytest.raises(RuntimeError):
        await adapter(["a", "b"])


async def test_gemini_embedding_returns_vectors_on_success():
    async def embed_content(model, contents, config):
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.3, 0.4]) for _ in contents]
        )

    adapter = _gemini_embedding(embed_content)

    assert await adapter(["a", "b"]) == [[0.3, 0.4], [0.3, 0.4]]
