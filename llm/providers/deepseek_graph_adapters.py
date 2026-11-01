"""Text-only LightRAG adapter; embeddings remain independently configured."""

from dataclasses import replace

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.profiles import LLMProfiles
from llm.providers.deepseek import DeepSeekProvider


class DeepSeekLLM:
    def __init__(self, model: str) -> None:
        self.provider = DeepSeekProvider()
        self.provider.model_name = model

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def __call__(self, prompt: str, **kwargs) -> str:
        messages: list[BaseMessage] = []
        if kwargs.get("system_prompt"):
            messages.append(SystemMessage(content=kwargs["system_prompt"]))
        roles = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        for item in kwargs.get("history_messages") or []:
            messages.append(roles[item["role"]](content=item["content"]))
        messages.append(HumanMessage(content=prompt))
        profile = LLMProfiles.GRAPH
        profile = replace(
            profile,
            max_tokens=kwargs.get("max_tokens", profile.max_tokens),
            temperature=kwargs.get("temperature", profile.temperature),
        )
        return await self.provider.generate(messages, profile=profile)
