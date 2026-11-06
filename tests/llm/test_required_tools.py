"""Обязательный tool действует один раунд, после результата возможен текст."""

from types import SimpleNamespace

from google.genai import types
from langchain_core.messages import HumanMessage

from llm.base import LLMUsage, ToolSpec
from llm.providers.gemini import GeminiProvider
from llm.providers.openai import OpenAIProvider


async def test_openai_required_tool_then_normal_generation(monkeypatch):
    provider = object.__new__(OpenAIProvider)
    requests = []

    async def stream(inputs, tools, temperature, max_tokens, timeout, on_delta, **kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return (
                "",
                [SimpleNamespace(call_id="scores-1", name="scores", arguments="{}")],
                LLMUsage(),
            )
        return "Ответ", [], LLMUsage()

    async def execute(name, args):
        return "63.27"

    monkeypatch.setattr(provider, "_run_responses_stream", stream)
    monkeypatch.setattr(provider, "_resolve_params", lambda profile: (0, 200, 10))
    result = await provider.generate_with_tools(
        [HumanMessage(content="А средний?")],
        tools=[ToolSpec("scores", "Баллы", {"type": "object", "properties": {}})],
        tool_executor=execute,
        required_tool="scores",
    )
    assert result.text == "Ответ"
    assert requests == [{"required_tool": "scores"}, {}]


async def test_gemini_required_tool_then_normal_generation(monkeypatch):
    provider = object.__new__(GeminiProvider)
    requests = []

    async def stream(contents, config, on_delta):
        requests.append(config.tool_config)
        if len(requests) == 1:
            call = types.FunctionCall(name="scores", args={})
            return "", [call], [types.Part(function_call=call)], LLMUsage()
        return "Ответ", [], [], LLMUsage()

    async def execute(name, args):
        return "63.27"

    monkeypatch.setattr(provider, "_run_gemini_stream", stream)
    monkeypatch.setattr(
        provider,
        "_build_request",
        lambda messages, profile: ([], types.GenerateContentConfig()),
    )
    result = await provider.generate_with_tools(
        [HumanMessage(content="А средний?")],
        tools=[ToolSpec("scores", "Баллы", {"type": "object", "properties": {}})],
        tool_executor=execute,
        required_tool="scores",
    )
    assert result.text == "Ответ"
    assert requests[0].function_calling_config.mode == "ANY"
    assert requests[0].function_calling_config.allowed_function_names == ["scores"]
    assert requests[1] is None
