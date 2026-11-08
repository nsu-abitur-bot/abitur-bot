"""Сбой графового LLM — ошибка, а не пустая строка (#290).

Пустая строка от адаптера на индексации заливает документ без сущностей и
связей (в hybrid он потом недостижим), а на запросе проходит дальше как
«контекста нет». Поэтому адаптер обязан падать, повторы делает LightRAG.
"""

from types import SimpleNamespace

import pytest

from llm.providers.gemini_graph_adapters import GeminiLLM
from llm.providers.openai_graph_adapters import OpenAILLM


class _Failing:
    async def __call__(self, *args, **kwargs):
        raise RuntimeError("429 Too Many Requests")


def _openai_llm(create) -> OpenAILLM:
    adapter = OpenAILLM(api_key="test-key")
    adapter.client = SimpleNamespace(  # type: ignore[assignment]
        responses=SimpleNamespace(create=create)
    )
    return adapter


def _gemini_llm(generate_content) -> GeminiLLM:
    adapter = GeminiLLM(api_key="test-key")
    adapter.client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    return adapter


async def test_openai_llm_raises_instead_of_empty_string():
    adapter = _openai_llm(_Failing())

    with pytest.raises(RuntimeError, match="429"):
        await adapter("Извлеки сущности из текста")


async def test_openai_llm_returns_text_on_success():
    async def create(model, input, max_output_tokens, temperature):
        return SimpleNamespace(output_text="  сущность; связь  ")

    adapter = _openai_llm(create)

    assert await adapter("текст") == "сущность; связь"


async def test_gemini_llm_raises_instead_of_empty_string():
    adapter = _gemini_llm(_Failing())

    with pytest.raises(RuntimeError, match="429"):
        await adapter("Извлеки сущности из текста")


async def test_gemini_llm_returns_text_on_success():
    async def generate_content(model, contents, config):
        return SimpleNamespace(text="  сущность; связь  ")

    adapter = _gemini_llm(generate_content)

    assert await adapter("текст") == "сущность; связь"
