"""Юнит-тесты поддержки инструментов: конвертация схемы + дефолтный fallback +
проброс инструмента в ask_local_llm (без обращения к сети и реальным LLM)."""

import asyncio

import pytest

from llm.base import BaseLLMProvider, LLMResult, ToolSpec
from llm.providers.gemini import _tools_to_gemini
from llm.providers.openai import _tool_spec_to_openai
from llm.tools import ADMISSION_SCORES_TOOL


def test_tool_spec_to_openai_format():
    result = _tool_spec_to_openai(ADMISSION_SCORES_TOOL)

    # Responses API: плоский формат функции (без вложенного "function").
    assert result["type"] == "function"
    assert result["name"] == "get_admission_scores"
    assert result["description"] == ADMISSION_SCORES_TOOL.description
    assert result["parameters"] is ADMISSION_SCORES_TOOL.parameters
    assert result["parameters"]["type"] == "object"
    assert "faculty" in result["parameters"]["properties"]


def test_tools_to_gemini_format():
    tool = _tools_to_gemini([ADMISSION_SCORES_TOOL])

    assert tool.function_declarations is not None
    assert len(tool.function_declarations) == 1
    declaration = tool.function_declarations[0]
    assert declaration.name == "get_admission_scores"
    assert declaration.description == ADMISSION_SCORES_TOOL.description
    # Схема передаётся как сырой JSON Schema.
    assert declaration.parameters_json_schema == ADMISSION_SCORES_TOOL.parameters


class _MinimalProvider(BaseLLMProvider):
    async def generate_with_usage(self, messages, profile=None, **kwargs) -> LLMResult:
        return LLMResult(text="обычный ответ")


@pytest.mark.asyncio
async def test_default_generate_with_tools_ignores_tools_non_stream():
    provider = _MinimalProvider()

    async def executor(name: str, args: dict) -> str:
        raise AssertionError("Инструмент не должен вызываться в дефолтном fallback")

    result = await provider.generate_with_tools(
        [], tools=[ADMISSION_SCORES_TOOL], tool_executor=executor
    )
    assert result.text == "обычный ответ"


@pytest.mark.asyncio
async def test_default_generate_with_tools_streams_via_on_delta():
    provider = _MinimalProvider()
    deltas: list[str] = []

    async def on_delta(chunk: str) -> None:
        deltas.append(chunk)

    async def executor(name: str, args: dict) -> str:
        return ""

    result = await provider.generate_with_tools(
        [],
        tools=[ADMISSION_SCORES_TOOL],
        tool_executor=executor,
        on_delta=on_delta,
    )
    # Дефолт делегирует стримингу -> одна финальная дельта.
    assert result.text == "обычный ответ"
    assert deltas == ["обычный ответ"]


class _ToolCallingProvider:
    """Провайдер, который сразу вызывает переданный tool_executor."""

    def __init__(self) -> None:
        self.tools_seen: list[ToolSpec] | None = None
        self.tool_result: str | None = None

    async def generate_with_tools(
        self, messages, tools, tool_executor, profile=None, on_delta=None
    ) -> LLMResult:
        self.tools_seen = tools
        self.tool_result = await tool_executor(
            "get_admission_scores", {"faculty": "ФИТ", "year": 2024}
        )
        return LLMResult(text=f"Проходной балл ФИТ — {self.tool_result}.")


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


@pytest.mark.asyncio
async def test_ask_local_llm_passes_admission_tool(monkeypatch):
    from llm import llm_client

    provider = _ToolCallingProvider()

    async def fake_get_redis_client() -> _FakeRedis:
        return _FakeRedis()

    async def fake_query_graph_with_sources(*args, **kwargs):
        return "Контекст из RAG", []

    async def fake_executor(name: str, args: dict) -> str:
        # Заменяем реальный исполнитель (с БД) на канонический ответ.
        assert name == "get_admission_scores"
        return "246 (бюджет, 2024)"

    monkeypatch.setattr(llm_client, "get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(llm_client, "get_abbrev_expander", lambda: _FakeAbbrevExpander())
    monkeypatch.setattr(llm_client, "get_faq_matcher", lambda: _FakeFaqMatcher())
    monkeypatch.setattr(
        llm_client, "query_graph_with_sources", fake_query_graph_with_sources
    )
    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(llm_client, "default_tool_executor", fake_executor)
    monkeypatch.setattr(llm_client, "_spawn_bg", lambda coro: coro.close())

    response = await llm_client.ask_local_llm(
        "Какой проходной балл на ФИТ в 2024?", "session-tool"
    )

    assert provider.tools_seen is not None
    assert provider.tools_seen[0].name == "get_admission_scores"
    assert provider.tool_result == "246 (бюджет, 2024)"
    assert "246" in response


if __name__ == "__main__":
    asyncio.run(test_default_generate_with_tools_streams_via_on_delta())
