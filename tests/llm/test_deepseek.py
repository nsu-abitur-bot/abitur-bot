"""Contract tests use the real OpenAI SDK over a synthetic HTTP transport."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIStatusError, AsyncOpenAI

from api.services import instance_settings
from config.instance import SPECS
from llm import factory
from llm.base import ToolSpec
from llm.profiles import LLMProfile
from llm.providers.deepseek import DeepSeekProvider
from llm.providers.deepseek_graph_adapters import DeepSeekLLM


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "contract-key")
    return DeepSeekProvider()


def completion(text="OK", finish="stop") -> dict[str, Any]:
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def sse(deltas, finish="stop", usage=True):
    chunks: list[dict[str, Any]] = [
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "delta": d, "finish_reason": None}],
        }
        for d in deltas
    ]
    chunks.append(
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        }
    )
    if usage:
        chunks.append(
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return httpx.Response(
        200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )


async def connect(provider, handler):
    await provider.client.close()
    provider.client = AsyncOpenAI(
        api_key="contract-key",
        base_url="https://deepseek.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


async def test_required_tool_is_forced_only_before_its_result(provider):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body["tool_choice"] == {
                "type": "function",
                "function": {"name": "scores"},
            }
            return sse(
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "forced-call",
                                "type": "function",
                                "function": {"name": "scores", "arguments": "{}"},
                            }
                        ]
                    }
                ],
                finish="tool_calls",
            )
        assert body["tool_choice"] == "auto"
        return sse([{"content": "Средний балл: 63.27"}])

    async def execute(name, args):
        return "Средний балл: 63.27"

    await connect(provider, handler)
    try:
        result = await provider.generate_with_tools(
            [HumanMessage(content="А средний?")],
            tools=[ToolSpec("scores", "Баллы", {"type": "object", "properties": {}})],
            tool_executor=execute,
            required_tool="scores",
        )
        assert result.text == "Средний балл: 63.27"
        assert len(requests) == 2
    finally:
        await provider.aclose()


async def test_chat_history_profile_and_usage(provider):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 42
        assert body["temperature"] == 0
        assert "input" not in body
        return httpx.Response(200, json=completion())

    await connect(provider, handler)
    try:
        result = await provider.generate_with_usage(
            [
                SystemMessage("rules"),
                HumanMessage("old"),
                AIMessage("answer"),
                HumanMessage("follow-up"),
            ],
            profile=LLMProfile(max_tokens=42, temperature=0, timeout=7),
        )
        assert result.text == "OK"
        assert result.usage.total_tokens == 12
        assert provider.get_embeddings_model() is None
    finally:
        await provider.aclose()


async def test_fragmented_multiple_tool_calls_and_stream_usage(provider):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body["tools"][0]["function"]["name"] == "scores"
            return sse(
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "type": "function",
                                "function": {"name": "scores", "arguments": '{"year":'},
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "2024}"}},
                            {
                                "index": 1,
                                "id": "b",
                                "type": "function",
                                "function": {"name": "scores", "arguments": "{}"},
                            },
                        ]
                    },
                ],
                finish="tool_calls",
            )
        assert body["messages"][-2:] == [
            {"role": "tool", "tool_call_id": "a", "content": "246"},
            {"role": "tool", "tool_call_id": "b", "content": "246"},
        ]
        return sse([{"content": "Балл "}, {"content": "246"}])

    await connect(provider, handler)
    executed, deltas = [], []

    async def execute(name, arguments):
        executed.append((name, arguments))
        return "246"

    async def delta(text):
        deltas.append(text)

    try:
        result = await provider.generate_with_tools(
            [HumanMessage("scores?")],
            [ToolSpec("scores", "scores", {"type": "object"})],
            execute,
            on_delta=delta,
        )
        assert result.text == "Балл 246"
        assert deltas == ["Балл ", "246"]
        assert executed == [("scores", {"year": 2024}), ("scores", {})]
        assert result.usage.total_tokens == 24
    finally:
        await provider.aclose()


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
async def test_incomplete_stream_fails(provider, finish):
    await connect(provider, lambda r: sse([{"content": "partial"}], finish=finish))
    try:
        with pytest.raises(RuntimeError, match="не завершил"):
            await provider.generate_stream_with_usage([HumanMessage("q")])
    finally:
        await provider.aclose()


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_transport_failure_propagates(provider, status):
    await connect(
        provider, lambda r: httpx.Response(status, json={"error": {"message": "failed"}})
    )
    try:
        with pytest.raises(APIStatusError):
            await provider.generate([HumanMessage("q")])
    finally:
        await provider.aclose()


async def test_graph_history_and_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "contract-key")
    adapter = DeepSeekLLM(model="graph-model")

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "graph-model"
        assert [m["content"] for m in body["messages"]] == [
            "rules",
            "old",
            "answer",
            "extract",
        ]
        return httpx.Response(200, json=completion("entities"))

    await connect(adapter.provider, handler)
    try:
        assert (
            await adapter(
                "extract",
                system_prompt="rules",
                history_messages=[
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
            == "entities"
        )
    finally:
        await adapter.aclose()


async def test_connection_check_uses_chat_contract_and_sanitizes(monkeypatch):
    values = {k: s.default for k, s in SPECS.items()}
    values["DEEPSEEK_API_KEY"] = "contract-key"

    def handler(request):
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(200, json=completion())

    original = AsyncOpenAI

    def client(**kwargs):
        return original(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(instance_settings, "AsyncOpenAI", client)
    for target in ("deepseek:chat", "deepseek:graph"):
        assert "проверены" in await instance_settings.check_connection(target, values)
    with pytest.raises(instance_settings.ConfigurationError):
        await instance_settings.check_connection("deepseek:vision", values)


async def test_factory_key_rotation_and_rollback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "contract-key")
    monkeypatch.setattr(factory, "_provider_instance", None)
    first = factory.get_llm_provider()
    assert isinstance(first, DeepSeekProvider)
    assert first is factory.get_llm_provider()
    monkeypatch.setenv("DEEPSEEK_MODEL", "other-model")
    second = factory.get_llm_provider()
    assert isinstance(second, DeepSeekProvider)
    assert second is not first
    assert second.model_name == "other-model"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "contract-gemini-key")
    assert type(factory.get_llm_provider()).__name__ == "GeminiProvider"
    await first.aclose()
    await second.aclose()


def test_assignments_are_text_only():
    assert "deepseek" in SPECS["LLM_PROVIDER"].choices
    assert "deepseek" in SPECS["LIGHTRAG_LLM_PROVIDER"].choices
    for key in (
        "FAQ_EMBEDDING_PROVIDER",
        "LIGHTRAG_EMBEDDING_PROVIDER",
        "PDF_PARSER_PROVIDER",
    ):
        assert "deepseek" not in SPECS[key].choices
    assert SPECS["DEEPSEEK_API_KEY"].secret
    values = {k: s.default for k, s in SPECS.items()}
    values.update(
        LLM_PROVIDER="deepseek",
        LIGHTRAG_LLM_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="contract-key",
    )
    assert instance_settings.required_checks({"DEEPSEEK_MODEL": "new"}, values) == {
        "deepseek:chat",
        "deepseek:graph",
    }


async def test_product_tool_loop_contract(provider, monkeypatch):
    from pipeline import llm_client

    monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["tools"][0]["function"]["name"] == "get_admission_scores"
        if len(requests) == 1:
            return sse(
                [
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "score-call",
                                "type": "function",
                                "function": {
                                    "name": "get_admission_scores",
                                    "arguments": '{"year":2024}',
                                },
                            }
                        ]
                    }
                ],
                finish="tool_calls",
            )
        assert body["messages"][-1]["content"] == "synthetic score: 246"
        return sse([{"content": "Тестовый балл: 246"}])

    executed, updates = [], []

    async def execute(name, arguments):
        executed.append((name, arguments))
        from pipeline.tools.admission_scores import AdmissionScoresResult

        return AdmissionScoresResult("synthetic score: 246", status="ok")

    async def update(text):
        updates.append(text)

    await connect(provider, handler)
    try:
        text, usage, name = await llm_client._run_tool_loop(
            [HumanMessage("Какой проходной балл на ФИТ в 2024 году?")],
            "contract:deepseek",
            stream_callback=update,
            tool_executor=execute,
        )
        assert executed == [("get_admission_scores", {"year": 2024})]
        assert updates[-1] == text == "Тестовый балл: 246"
        assert usage.total_tokens == 24
        assert name == "DeepSeekProvider"
    finally:
        await provider.aclose()


async def test_tool_round_limit_forces_final_answer(provider):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 4:
            assert body["tool_choice"] == "none"
            return sse([{"content": "done"}])
        return sse(
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"call-{len(requests)}",
                            "type": "function",
                            "function": {
                                "name": "scores",
                                "arguments": "{}",
                            },
                        }
                    ]
                }
            ],
            finish="tool_calls",
        )

    async def execute(name, arguments):
        return "result"

    await connect(provider, handler)
    try:
        result = await provider.generate_with_tools(
            [HumanMessage("q")],
            [ToolSpec("scores", "scores", {"type": "object"})],
            execute,
        )
        assert result.text == "done"
        assert result.usage.total_tokens == 48
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "name,arguments", [("unknown", "{}"), ("scores", "{"), ("scores", "[]")]
)
async def test_invalid_tool_call_never_executes(provider, name, arguments):
    await connect(
        provider,
        lambda r: sse(
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "a",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                },
            ],
            finish="tool_calls",
        ),
    )

    async def execute(name, arguments):
        pytest.fail("Invalid tool call must not execute")

    try:
        with pytest.raises(ValueError):
            await provider.generate_with_tools(
                [HumanMessage("q")],
                [ToolSpec("scores", "scores", {"type": "object"})],
                execute,
            )
    finally:
        await provider.aclose()


def test_faq_embeddings_do_not_inherit_deepseek(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("FAQ_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "contract-gemini-key")
    monkeypatch.setattr(factory, "_faq_provider", None)
    assert type(factory.get_faq_provider()).__name__ == "GeminiProvider"


def test_live_comparison_scenarios_are_valid_and_not_offline_evidence():
    from evals.schema import load_scenarios

    scenarios = load_scenarios("evals/deepseek_scenarios.yaml")
    assert len(scenarios) == 3
    assert all(s.requires_llm for s in scenarios)
    assert all(s.approved_by is None for s in scenarios)
