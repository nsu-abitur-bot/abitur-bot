"""Сбой поиска по базе знаний не должен выглядеть как «информации нет» (#282).

Пустой результат и упавший поиск — разные состояния: на пустом модель честно
отвечает «не нашёл», на упавшем пользователь должен узнать, что сервис
временно недоступен, а в логах должна остаться ошибка.
"""

import logging
from types import SimpleNamespace

import pytest

from llm.base import LLMResult
from pipeline import llm_client
from pipeline.prompts import RAG_CONTEXT_NOT_FOUND, RAG_UNAVAILABLE_ANSWER
from rag.retriever import RetrievalError


def _close_background_coro(coro) -> None:
    coro.close()


class _RecordingProvider:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        self.system_prompts.append(str(messages[0].content))
        return LLMResult(text="Ответ модели")


@pytest.fixture
def pipeline_without_io(monkeypatch):
    async def crag_disabled():
        return SimpleNamespace(enabled=False)

    provider = _RecordingProvider()
    monkeypatch.setattr(llm_client, "load_crag_config", crag_disabled)
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
    return provider


def _ctx() -> llm_client._Ctx:
    return llm_client._Ctx(session_id="s-1", user_id=0)


async def _answer_with_retrieval(monkeypatch, retrieve) -> str:
    monkeypatch.setattr(llm_client, "query_graph_with_sources", retrieve)
    ctx = _ctx()
    question = "Когда приём?"
    rag = await llm_client._retrieve_context(question, "", ctx)
    text, _ = await llm_client._generate_answer(rag, [], question, ctx)
    return text


async def test_retrieval_error_tells_user_service_is_unavailable(
    monkeypatch, pipeline_without_io, caplog
):
    async def failing_retrieval(query, conversation_history=None):
        raise RetrievalError("RateLimitError 429")

    with caplog.at_level(logging.ERROR, logger="pipeline.llm_client"):
        text = await _answer_with_retrieval(monkeypatch, failing_retrieval)

    assert text == RAG_UNAVAILABLE_ANSWER
    # Модель не зовём: с пустым контекстом она сказала бы «не нашёл в базе».
    assert pipeline_without_io.system_prompts == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("429" in r.getMessage() for r in errors)


async def test_unexpected_retrieval_crash_is_also_unavailable(
    monkeypatch, pipeline_without_io
):
    async def crashing_retrieval(query, conversation_history=None):
        raise ValueError("broken store")

    text = await _answer_with_retrieval(monkeypatch, crashing_retrieval)

    assert text == RAG_UNAVAILABLE_ANSWER
    assert pipeline_without_io.system_prompts == []


async def test_empty_retrieval_still_means_not_found(monkeypatch, pipeline_without_io):
    async def empty_retrieval(query, conversation_history=None):
        return "", []

    text = await _answer_with_retrieval(monkeypatch, empty_retrieval)

    assert text == "Ответ модели"
    [system_prompt] = pipeline_without_io.system_prompts
    assert RAG_CONTEXT_NOT_FOUND in system_prompt


async def test_found_context_goes_to_model(monkeypatch, pipeline_without_io):
    async def found_retrieval(query, conversation_history=None):
        return "Приём документов с 20 июня.", []

    text = await _answer_with_retrieval(monkeypatch, found_retrieval)

    assert text == "Ответ модели"
    [system_prompt] = pipeline_without_io.system_prompts
    assert "Приём документов с 20 июня." in system_prompt
    assert RAG_CONTEXT_NOT_FOUND not in system_prompt
