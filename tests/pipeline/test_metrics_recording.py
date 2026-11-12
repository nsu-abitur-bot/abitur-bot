"""Запись метрик наблюдаемости в лог ответов (#242).

Пайплайн кладёт latency_ms / no_answer / error в message_metadata записи
llm_response — из них SQL собирает агрегаты по периодам для дашборда.
"""

import asyncio

from llm.base import LLMResult
from pipeline import llm_client
from rag.trace import RetrievalResult


class _FakeRedis:
    async def add_message(self, session_id: str, message: dict) -> None:
        return None

    async def get_history(self, session_id: str) -> list[dict]:
        return []


class _FakeAbbrevExpander:
    def expand(self, message: str) -> str:
        return message


class _NoFaqMatcher:
    async def match_async(self, user_question: str) -> str | None:
        return None


class _Provider:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    async def generate_with_usage(self, messages, profile) -> LLMResult:
        if self._error:
            raise self._error
        return LLMResult(text=self._text or "")

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        return await self.generate_with_usage(messages, profile)


async def _get_fake_redis() -> _FakeRedis:
    return _FakeRedis()


async def _fake_query(query, conversation_history=None, search_query=None):
    return RetrievalResult(context="Контекст из базы знаний")


def _setup(monkeypatch, provider: _Provider) -> list[dict]:
    saved: list[dict] = []

    async def fake_save(**kwargs) -> None:
        saved.append(kwargs)

    async def fake_classify(message, log_entry_id, session_id) -> None:
        return None

    monkeypatch.setattr(llm_client, "get_redis_client", _get_fake_redis)
    monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: _FakeAbbrevExpander())
    monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: _NoFaqMatcher())
    monkeypatch.setattr(llm_client, "query_graph_with_sources", _fake_query)
    monkeypatch.setattr(llm_client, "query_graph_with_crag", _fake_query)
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(llm_client, "_save_log_to_db", fake_save)
    monkeypatch.setattr(llm_client, "_classify_intent_bg", fake_classify)
    return saved


def _llm_responses(saved: list[dict]) -> list[dict]:
    return [e for e in saved if e["message_type"] == "llm_response"]


async def test_answer_records_latency(monkeypatch):
    saved = _setup(monkeypatch, _Provider(text="Документы принимают с 20 июня."))

    await llm_client.ask_local_llm("Когда подать документы?", "s-metrics")
    await asyncio.sleep(0)

    (entry,) = _llm_responses(saved)
    assert isinstance(entry["message_metadata"]["latency_ms"], float)
    assert entry["message_metadata"]["latency_ms"] >= 0
    assert entry["message_metadata"]["no_answer"] is False
    assert "error" not in entry["message_metadata"]


async def test_refusal_records_no_answer_flag(monkeypatch):
    saved = _setup(monkeypatch, _Provider(text="Не нашёл информации по вопросу."))

    await llm_client.ask_local_llm("Есть ли общежитие?", "s-metrics-refusal")
    await asyncio.sleep(0)

    (entry,) = _llm_responses(saved)
    assert entry["message_metadata"]["no_answer"] is True


async def test_provider_error_records_error_flag(monkeypatch):
    saved = _setup(monkeypatch, _Provider(error=RuntimeError("provider is down")))

    answer = await llm_client.ask_local_llm("Когда подать документы?", "s-metrics-err")
    await asyncio.sleep(0)

    (entry,) = _llm_responses(saved)
    assert answer.text == "LLM временно недоступна."
    assert entry["message_metadata"]["error"] is True
    assert entry["message_metadata"]["latency_ms"] >= 0
