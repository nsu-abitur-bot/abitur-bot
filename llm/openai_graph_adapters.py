import logging
from typing import List

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Размерности для стандартных моделей эмбеддингов OpenAI
OPENAI_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAILLM:
    """LLM-адаптер для LightRAG на базе OpenAI."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            messages = []
            if kwargs.get("system_prompt"):
                messages.append({"role": "system", "content": kwargs["system_prompt"]})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.2),
            )
            content = response.choices[0].message.content
            return content.strip() if isinstance(content, str) else str(content or "")
        except Exception as e:
            logger.error(f"Error in OpenAILLM: {e}")
            return ""


class OpenAIEmbedding:
    """Embedding-адаптер для LightRAG на базе OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self.embedding_dim = OPENAI_EMBEDDING_DIMS.get(model, 1536)

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error in OpenAIEmbedding: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]
