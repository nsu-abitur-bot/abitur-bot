import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

import pytest

from llm.base import LLMResult, LLMUsage
from llm.llm_client import ask_local_llm


class _FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    async def add_message(self, session_id: str, message: dict) -> None:
        self.messages.append((session_id, message))

    async def get_history(self, session_id: str) -> list[dict]:
        return []


class _FakeAbbrevExpander:
    def expand(self, text: str) -> str:
        return text


class _FakeFaqMatcher:
    async def match_async(self, text: str) -> str | None:
        return None


class _FakeStreamingProvider:
    async def generate_stream_with_usage(self, messages, profile=None, on_delta=None):
        if on_delta is not None:
            await on_delta("Ответ")
            await on_delta(" бота")
        return LLMResult(
            text="Ответ бота",
            usage=LLMUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
        )


@pytest.mark.asyncio
async def test_streaming_llm_usage_is_saved(monkeypatch):
    saved_logs = []
    background_tasks: list[asyncio.Task] = []
    streamed_updates = []

    async def fake_get_redis_client() -> _FakeRedis:
        return _FakeRedis()

    async def fake_query_graph_with_sources(*args, **kwargs):
        logging.getLogger("lightrag").info(
            "Query nodes: IT, факультеты (top_k:40, cosine:0.2)"
        )
        logging.getLogger("lightrag").warning(
            "Rerank is enabled but no rerank model is configured"
        )
        return "Контекст", []

    async def fake_save_log_to_db(**kwargs) -> None:
        saved_logs.append(kwargs)

    async def fake_classify_intent_bg(*args, **kwargs) -> None:
        return None

    async def stream_callback(content: str) -> None:
        streamed_updates.append(content)

    def spawn_bg(coro: Coroutine[Any, Any, None]) -> None:
        background_tasks.append(asyncio.create_task(coro))

    monkeypatch.setattr("llm.llm_client.get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(
        "llm.llm_client.get_abbrev_expander", lambda: _FakeAbbrevExpander()
    )
    monkeypatch.setattr("llm.llm_client.get_faq_matcher", lambda: _FakeFaqMatcher())
    monkeypatch.setattr(
        "llm.llm_client.query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(
        "llm.llm_client.get_llm_provider", lambda: _FakeStreamingProvider()
    )
    monkeypatch.setattr("llm.llm_client._save_log_to_db", fake_save_log_to_db)
    monkeypatch.setattr("llm.llm_client._classify_intent_bg", fake_classify_intent_bg)
    monkeypatch.setattr("llm.llm_client._spawn_bg", spawn_bg)

    response = await ask_local_llm(
        "Когда прием?",
        session_id="test-session",
        user_id=1,
        stream_callback=stream_callback,
    )
    if background_tasks:
        await asyncio.gather(*background_tasks)

    assert response == "Ответ бота"
    assert streamed_updates == ["Ответ", "Ответ бота"]

    llm_logs = [log for log in saved_logs if log["message_type"] == "llm_response"]
    assert len(llm_logs) == 1
    assert llm_logs[0]["tokens_used"] == 50
    assert llm_logs[0]["message_metadata"]["tokens"] == {
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
    }

    rag_query_logs = [
        log for log in saved_logs if log["message_type"] == "rag_query"
    ]
    rag_response_logs = [
        log for log in saved_logs if log["message_type"] == "rag_response"
    ]
    assert len(rag_query_logs) == 1
    assert len(rag_response_logs) == 1
    assert "Когда прием?" in rag_query_logs[0]["content"]
    assert rag_query_logs[0]["message_metadata"]["title"] == "Запрос к базе знаний"
    assert "Трассировка поиска в базе знаний" in rag_response_logs[0]["content"]
    assert "INFO: Query nodes: IT, факультеты" in rag_response_logs[0]["content"]
    assert "WARNING: Rerank is enabled" in rag_response_logs[0]["content"]
    assert "Ответ базы знаний:\nКонтекст" in rag_response_logs[0]["content"]
    assert rag_response_logs[0]["message_metadata"]["title"] == "Ответ базы знаний"
    assert rag_response_logs[0]["message_metadata"]["internal_logs_count"] == 2
