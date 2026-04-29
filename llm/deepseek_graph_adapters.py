import logging
import os

from openai import AsyncOpenAI

from llm.profiles import LLMProfiles

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY не установлен.")

        self.model = model
        self.profile = LLMProfiles.GRAPH

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=float(self.profile.timeout or 120),
        )
        logger.info(
            "DeepSeekLLM initialized (model=%s, timeout=%ss)",
            self.model,
            self.profile.timeout,
        )

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            messages: list[dict[str, str]] = []
            if "system_prompt" in kwargs and kwargs["system_prompt"]:
                messages.append({"role": "system", "content": kwargs["system_prompt"]})
            messages.append({"role": "user", "content": prompt})

            temperature: float = float(
                kwargs.get("temperature", self.profile.temperature or 0.7)
            )
            max_tokens: int = int(
                kwargs.get("max_tokens", self.profile.max_tokens or 2000)
            )

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = completion.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            return str(content or "")
        except Exception as e:
            logger.error(
                "Error in DeepSeekLLM (model=%s): %s — %s",
                self.model,
                type(e).__name__,
                e,
            )
            return ""
