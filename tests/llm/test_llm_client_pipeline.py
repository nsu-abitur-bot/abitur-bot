import asyncio

from llm import llm_client
from llm.base import LLMResult


class _FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    async def add_message(self, session_id: str, message: dict) -> None:
        self.messages.append((session_id, message))

    async def get_history(self, session_id: str) -> list[dict]:
        return [
            message
            for saved_session_id, message in self.messages
            if saved_session_id == session_id
        ]


class _FakeAbbrevExpander:
    def expand(self, message: str) -> str:
        return message


class _FakeProvider:
    async def generate_with_usage(self, messages, profile) -> LLMResult:
        return LLMResult(text="Ответ из LLM")


def _close_background_coro(coro) -> None:
    coro.close()


async def _get_fake_redis() -> _FakeRedis:
    return _FakeRedis()


async def test_rag_does_not_start_before_faq_matcher_finishes(monkeypatch):
    faq_done = False

    class FakeFaqMatcher:
        async def match_async(self, user_question: str) -> str | None:
            nonlocal faq_done
            await asyncio.sleep(0.01)
            faq_done = True
            return None

    async def fake_query_graph_with_sources(query: str, conversation_history=None):
        assert faq_done
        return "Контекст из RAG", []

    monkeypatch.setattr(llm_client, "get_redis_client", _get_fake_redis)
    monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: _FakeAbbrevExpander())
    monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: FakeFaqMatcher())
    monkeypatch.setattr(
        llm_client, "query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)

    response = await llm_client.ask_local_llm("Когда подать документы?", "session-1")

    assert response == "Ответ из LLM"


async def test_faq_hit_returns_without_rag(monkeypatch):
    rag_started = False

    class FakeFaqMatcher:
        async def match_async(self, user_question: str) -> str | None:
            return "Ответ из FAQ"

    async def fake_query_graph_with_sources(query: str, conversation_history=None):
        nonlocal rag_started
        rag_started = True
        return "Контекст из RAG", []

    monkeypatch.setattr(llm_client, "get_redis_client", _get_fake_redis)
    monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: _FakeAbbrevExpander())
    monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: FakeFaqMatcher())
    monkeypatch.setattr(
        llm_client, "query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)

    response = await llm_client.ask_local_llm("Какие факультеты есть?", "session-2")

    assert response == "Ответ из FAQ"
    assert not rag_started
