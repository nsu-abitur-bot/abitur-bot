"""DeepSeek Chat Completions transport; text generation only."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI

from config.instance import get_setting
from llm.base import BaseLLMProvider, LLMResult, LLMUsage, ToolExecutor, ToolSpec
from llm.profiles import LLMProfile


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self) -> None:
        key = get_setting("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY не установлен")
        self.model_name = get_setting("DEEPSEEK_MODEL", "deepseek-flash")
        self.client = AsyncOpenAI(
            api_key=key,
            base_url=get_setting("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            timeout=float(get_setting("DEEPSEEK_TIMEOUT_SECONDS", "60")),
        )

    async def aclose(self) -> None:
        await self.client.close()

    def _params(self, profile: LLMProfile | None) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": profile.temperature
            if profile and profile.temperature is not None
            else float(get_setting("DEEPSEEK_TEMPERATURE", "0.2")),
            "max_tokens": profile.max_tokens
            if profile and profile.max_tokens is not None
            else int(get_setting("DEEPSEEK_MAX_TOKENS", "1024")),
            "timeout": profile.timeout
            if profile and profile.timeout is not None
            else float(get_setting("DEEPSEEK_TIMEOUT_SECONDS", "60")),
            # The API defaults to thinking. Keep latency/token budgets and the
            # existing text-only history contract explicit for this integration.
            "extra_body": {"thinking": {"type": "disabled"}},
        }

    @staticmethod
    def _messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        result = []
        for message in messages:
            if message.type not in roles or not isinstance(message.content, str):
                raise ValueError("DeepSeek принимает только текстовую историю чата")
            result.append({"role": roles[message.type], "content": message.content})
        return result

    async def generate_with_usage(
        self, messages: list[BaseMessage], profile: LLMProfile | None = None, **kwargs
    ) -> LLMResult:
        response = await self.client.chat.completions.create(
            messages=cast(Any, self._messages(messages)), **self._params(profile)
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise RuntimeError(
                "DeepSeek не завершил текстовый ответ: "
                f"finish_reason={choice.finish_reason}, "
                f"completion_tokens={_usage(response.usage).completion_tokens}, "
                f"max_tokens={self._params(profile)['max_tokens']}"
            )
        return LLMResult(choice.message.content or "", _usage(response.usage))

    async def generate_stream_with_usage(
        self,
        messages: list[BaseMessage],
        profile: LLMProfile | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        return await self.generate_with_tools(
            messages, [], _unused_executor, profile=profile, on_delta=on_delta
        )

    async def generate_stream(
        self, messages: list[BaseMessage], profile: LLMProfile | None = None, **kwargs
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            messages=cast(Any, self._messages(messages)),
            stream=True,
            **self._params(profile),
        )
        finish = None
        async with stream:
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    finish = choice.finish_reason or finish
                    if choice.delta.content:
                        yield choice.delta.content
        if finish != "stop":
            raise RuntimeError("DeepSeek не завершил потоковый ответ")

    async def generate_with_tools(
        self,
        messages: list[BaseMessage],
        tools: list[ToolSpec],
        tool_executor: ToolExecutor,
        profile: LLMProfile | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs,
    ) -> LLMResult:
        history = self._messages(messages)
        usage = LLMUsage()
        for iteration in range(4):
            params = self._params(profile)
            if tools:
                params["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    }
                    for tool in tools
                ]
                params["tool_choice"] = "none" if iteration == 3 else "auto"
            stream = await self.client.chat.completions.create(
                messages=cast(Any, history),
                stream=True,
                stream_options={"include_usage": True},
                **params,
            )
            text: list[str] = []
            calls: dict[int, dict[str, Any]] = {}
            finish = None
            async with stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        part = _usage(chunk.usage)
                        usage.prompt_tokens += part.prompt_tokens
                        usage.completion_tokens += part.completion_tokens
                        usage.total_tokens += part.total_tokens
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish = choice.finish_reason or finish
                    delta = choice.delta
                    if delta.content:
                        text.append(delta.content)
                        if on_delta is not None:
                            await on_delta(delta.content)
                    for call in delta.tool_calls or []:
                        item = calls.setdefault(
                            call.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        item["id"] += call.id or ""
                        if call.function:
                            item["function"]["name"] += call.function.name or ""
                            item["function"]["arguments"] += call.function.arguments or ""
            if not calls:
                if finish != "stop":
                    raise RuntimeError("DeepSeek не завершил потоковый ответ")
                return LLMResult("".join(text).strip(), usage)
            if finish != "tool_calls" or iteration == 3:
                raise RuntimeError("DeepSeek не завершил вызов инструментов")
            history.append(
                {
                    "role": "assistant",
                    "content": "".join(text) or None,
                    "tool_calls": list(calls.values()),
                }
            )
            for call in calls.values():
                function = call["function"]
                if not call["id"] or function["name"] not in {t.name for t in tools}:
                    raise ValueError("DeepSeek вернул неизвестный инструмент")
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Аргументы инструмента должны быть объектом")
                output = await tool_executor(function["name"], arguments)
                history.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )
        raise RuntimeError("Превышен лимит вызовов инструментов DeepSeek")


def _usage(raw: Any) -> LLMUsage:
    if raw is None:
        return LLMUsage()
    return LLMUsage(raw.prompt_tokens, raw.completion_tokens, raw.total_tokens)


async def _unused_executor(name: str, arguments: dict) -> str:
    raise ValueError("Инструменты не переданы")
