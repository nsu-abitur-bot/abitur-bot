import asyncio
import logging

from langchain_core.messages import HumanMessage
from redis.exceptions import ConnectionError as RedisConnectionError

from llm.base import LLMResult
from pipeline import llm_client


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

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        return await self.generate_with_usage(messages, profile)


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
    # Мокаем ОБЕ ветки ретрива: какая именно выполнится, зависит от CRAG_ENABLED,
    # а тест проверяет порядок FAQ→RAG и не должен зависеть от этой настройки.
    monkeypatch.setattr(
        llm_client, "query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(
        llm_client, "query_graph_with_crag", fake_query_graph_with_sources
    )
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)

    response = await llm_client.ask_local_llm("Когда подать документы?", "session-1")

    assert response.text == "Ответ из LLM"


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
    # Обе ветки ретрива — иначе assert not rag_started прошёл бы вхолостую,
    # даже если бы RAG реально сходил через CRAG-ветку.
    monkeypatch.setattr(
        llm_client, "query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(
        llm_client, "query_graph_with_crag", fake_query_graph_with_sources
    )
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)

    response = await llm_client.ask_local_llm("Какие факультеты есть?", "session-2")

    assert response.text == "Ответ из FAQ"
    assert not rag_started


class _NoFaq:
    async def match_async(self, user_question: str) -> str | None:
        return None


class _RecordingProvider:
    """Запоминает сообщения, ушедшие в модель."""

    def __init__(self) -> None:
        self.messages: list = []

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        self.messages = list(messages)
        return LLMResult(text="Ответ из LLM")


def _patch_pipeline(monkeypatch, redis, provider) -> None:
    async def get_redis():
        return redis

    async def fake_query(query: str, conversation_history=None):
        return "Контекст из RAG", []

    monkeypatch.setattr(llm_client, "get_redis_client", get_redis)
    monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: _FakeAbbrevExpander())
    monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: _NoFaq())
    monkeypatch.setattr(llm_client, "query_graph_with_sources", fake_query)
    monkeypatch.setattr(llm_client, "query_graph_with_crag", fake_query)
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(llm_client, "_spawn_bg", _close_background_coro)


class _EmptyHistoryRedis(_FakeRedis):
    """История потеряна: TTL истёк, Redis перезапущен или get_history заглушил сбой."""

    async def get_history(self, session_id: str) -> list[dict]:
        return []


class _BrokenRedis:
    """Redis недоступен: и запись, и чтение падают."""

    async def add_message(self, session_id: str, message: dict) -> None:
        raise RedisConnectionError("Connection refused")

    async def get_history(self, session_id: str) -> list[dict]:
        raise RedisConnectionError("Connection refused")


async def test_question_reaches_model_with_empty_history(monkeypatch):
    """#280: без истории модель раньше получала один системный промпт без вопроса."""
    provider = _RecordingProvider()
    _patch_pipeline(monkeypatch, _EmptyHistoryRedis(), provider)

    response = await llm_client.ask_local_llm("Когда подать документы?", "s-empty")

    assert response.text == "Ответ из LLM"
    assert isinstance(provider.messages[-1], HumanMessage)
    assert provider.messages[-1].content == "Когда подать документы?"


async def test_redis_down_still_answers(monkeypatch, caplog):
    """#281: недоступный Redis не превращает ответ в «Что-то пошло не так»."""
    provider = _RecordingProvider()
    _patch_pipeline(monkeypatch, _BrokenRedis(), provider)

    with caplog.at_level(logging.WARNING, logger=llm_client.logger.name):
        response = await llm_client.ask_local_llm("Когда подать документы?", "s-down")

    assert response.text == "Ответ из LLM"
    assert [m.type for m in provider.messages] == ["system", "human"]
    assert provider.messages[-1].content == "Когда подать документы?"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("add_message" in r.getMessage() for r in warnings)
    assert any("get_history" in r.getMessage() for r in warnings)


async def test_history_does_not_duplicate_current_question(monkeypatch):
    """Вопрос пишется в Redis до чтения истории — в промпте он должен быть один раз."""
    redis = _FakeRedis()
    session_id = "s-history"
    await redis.add_message(session_id, {"role": "user", "content": "Есть ли общежитие?"})
    await redis.add_message(session_id, {"role": "assistant", "content": "Да, есть."})
    provider = _RecordingProvider()
    _patch_pipeline(monkeypatch, redis, provider)

    await llm_client.ask_local_llm("Сколько оно стоит?", session_id)

    assert [m.type for m in provider.messages] == ["system", "human", "ai", "human"]
    assert [m.content for m in provider.messages[1:]] == [
        "Есть ли общежитие?",
        "Да, есть.",
        "Сколько оно стоит?",
    ]


async def test_repeated_question_keeps_earlier_turn(monkeypatch):
    """Тот же вопрос раньше в переписке — часть истории, вырезать его нельзя."""
    redis = _FakeRedis()
    session_id = "s-repeat"
    await redis.add_message(session_id, {"role": "user", "content": "Есть ли общежитие?"})
    await redis.add_message(session_id, {"role": "assistant", "content": "Да, есть."})
    provider = _RecordingProvider()
    _patch_pipeline(monkeypatch, redis, provider)

    await llm_client.ask_local_llm("Есть ли общежитие?", session_id)

    assert [m.content for m in provider.messages[1:]] == [
        "Есть ли общежитие?",
        "Да, есть.",
        "Есть ли общежитие?",
    ]
