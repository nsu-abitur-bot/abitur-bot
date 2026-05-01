import asyncio
import logging
import os

from llm.profiles import LLMProfiles

logger = logging.getLogger(__name__)

# Cerebras free-tier limits: 30 req/min, 900 req/hr
_RATE_LIMIT_WAIT_SECONDS = 60


def _load_cerebras_async_class():
    try:
        from cerebras.cloud.sdk import AsyncCerebras

        return AsyncCerebras
    except ImportError as exc:
        raise RuntimeError(
            "Пакет cerebras-cloud-sdk не установлен. "
            "Установите его: pip install cerebras-cloud-sdk"
        ) from exc


class CerebrasLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-oss-120b",
        top_p: float = 1.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY не установлен.")

        self.model = model
        self.top_p = top_p
        self.max_retries = max_retries
        
        self.profile = LLMProfiles.GRAPH

        AsyncCerebras = _load_cerebras_async_class()
        # Let the SDK do its own retries for transient errors (5xx, network).
        # We handle 429 rate-limit manually with longer pauses.
        self.client = AsyncCerebras(
            api_key=self.api_key,
            timeout=self.profile.timeout,
            max_retries=self.max_retries,
        )
        logger.info(
            "CerebrasLLM initialized (model=%s, timeout=%ss, retries=%s)",
            self.model,
            self.profile.timeout,
            self.max_retries,
        )

    async def __call__(self, prompt: str, **kwargs) -> str:
        messages = []
        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            messages.append({"role": "system", "content": kwargs["system_prompt"]})
        messages.append({"role": "user", "content": prompt})

        temperature = kwargs.get("temperature", self.profile.temperature)
        max_tokens = kwargs.get("max_tokens", self.profile.max_tokens)

        # Retry loop for rate-limit (429) errors
        max_rate_limit_retries = 5
        for attempt in range(1, max_rate_limit_retries + 1):
            try:
                completion = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    stream=False,
                    timeout=self.profile.timeout,
                )

                content = completion.choices[0].message.content  # type: ignore[union-attr]
                if isinstance(content, str):
                    return content.strip()
                return str(content or "")

            except Exception as e:
                error_name = type(e).__name__
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()

                if is_rate_limit and attempt < max_rate_limit_retries:
                    wait = _RATE_LIMIT_WAIT_SECONDS * attempt
                    logger.warning(
                        "CerebrasLLM rate-limited (attempt %d/%d). "
                        "Waiting %ds before retry…",
                        attempt,
                        max_rate_limit_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error(
                    "Error in CerebrasLLM (model=%s, attempt=%d): %s — %s",
                    self.model,
                    attempt,
                    error_name,
                    e,
                )
                return ""

        return ""
