"""Регистрация вопросов без ответа в очередь редактору (#258).

Отказ «не нашёл информации» — это пробел в наполнении базы знаний: вопрос
уходит в лог с типом no_answer и потом агрегируется с частотой. Сервисные
сбои (RAG недоступен, ошибка модели) в очередь попадать не должны — это
ошибки инфраструктуры, их место в метриках, а не в очереди редактора.
"""

import asyncio

from llm.base import LLMResult
from pipeline import llm_client
from rag.retriever import RetrievalError
from rag.trace import RetrievalResult


class _FakeRedis:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    async def add_message(self, session_id: str, message: dict) -> None:
        self.messages.append((session_id, message))

    async def get_history(self, session_id: str) -> list[dict]:
        return []


class _FakeAbbrevExpander:
    def expand(self, message: str) -> str:
        return message


class _NoFaqMatcher:
    async def match_async(self, user_question: str) -> str | None:
        return None


class _Provider:
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, messages, profile):
        if self._text and "не нашёл" in self._text.casefold():
            return '{"verdict":"refer"}'
        return '{"verdict":"supported","quotes":["Документы принимают с 20 июня."]}'

    async def generate_with_usage(self, messages, profile) -> LLMResult:
        return LLMResult(text=self._text)

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        return await self.generate_with_usage(messages, profile)


async def _get_fake_redis() -> _FakeRedis:
    return _FakeRedis()


async def _fake_query(query, conversation_history=None, search_query=None):
    evidence = "Документы принимают с 20 июня."
    return RetrievalResult(context=evidence, evidence=[evidence])


def _setup(monkeypatch, provider_text: str) -> list[dict]:
    """Общее окружение ask_local_llm; возвращает список записей лога."""
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
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: _Provider(provider_text))
    monkeypatch.setattr(llm_client, "_save_log_to_db", fake_save)
    monkeypatch.setattr(llm_client, "_classify_intent_bg", fake_classify)
    return saved


def _no_answer_entries(saved: list[dict]) -> list[dict]:
    return [entry for entry in saved if entry["message_type"] == "no_answer"]


async def test_refusal_registers_question(monkeypatch):
    saved = _setup(monkeypatch, "К сожалению, не нашёл информации по этому вопросу.")

    await llm_client.ask_local_llm("Есть ли стипендия для аспирантов?", "s-refusal")
    await asyncio.sleep(0)  # дорабатывают фоновые записи лога

    entries = _no_answer_entries(saved)
    assert len(entries) == 1
    assert entries[0]["content"] == "Есть ли стипендия для аспирантов?"


async def test_empty_answer_registers_question(monkeypatch):
    saved = _setup(monkeypatch, "")

    await llm_client.ask_local_llm("Как проходит апелляция?", "s-empty")
    await asyncio.sleep(0)  # дорабатывают фоновые записи лога

    assert len(_no_answer_entries(saved)) == 1


async def test_normal_answer_not_registered(monkeypatch):
    saved = _setup(monkeypatch, "Документы принимают с 20 июня.")

    await llm_client.ask_local_llm("Когда подать документы?", "s-normal")
    await asyncio.sleep(0)  # дорабатывают фоновые записи лога

    assert _no_answer_entries(saved) == []


async def test_rag_failure_not_registered(monkeypatch):
    saved = _setup(monkeypatch, "любой текст не используется")

    async def failing_query(query, conversation_history=None, search_query=None):
        raise RetrievalError("RAG is down")

    monkeypatch.setattr(llm_client, "query_graph_with_sources", failing_query)
    monkeypatch.setattr(llm_client, "query_graph_with_crag", failing_query)

    answer = await llm_client.ask_local_llm("Когда подать документы?", "s-rag-down")
    await asyncio.sleep(0)  # дорабатывают фоновые записи лога

    # Сбой сервиса — не пробел контента: в очереди редактора его быть не должно.
    assert _no_answer_entries(saved) == []
    assert answer.text == llm_client.RAG_UNAVAILABLE_ANSWER
