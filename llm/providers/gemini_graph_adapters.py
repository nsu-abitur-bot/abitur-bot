import logging
from typing import List

from google import genai
from google.genai import types

from llm.profiles import LLMProfiles

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_DIMS: dict[str, int] = {
    "gemini-embedding-2-preview": 768,
    "text-embedding-004": 768,
}


class GeminiLLM:
    """LLM-адаптер для LightRAG на базе Google Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.profile = LLMProfiles.GRAPH

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            # If there's a system prompt, we might need to handle it.
            # In google-genai, system instructions can be passed via config.
            system_prompt = kwargs.get("system_prompt")
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.profile.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.profile.max_tokens),
            )

            if system_prompt:
                config.system_instruction = system_prompt

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            content = response.text
            return content.strip() if isinstance(content, str) else str(content or "")
        except Exception as e:
            logger.error(f"Error in GeminiLLM: {e}")
            return ""


class GeminiEmbedding:
    """Embedding-адаптер для LightRAG на базе Google Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-2-preview") -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.embedding_dim = GEMINI_EMBEDDING_DIMS.get(model, 768)
        self.profile = LLMProfiles.EMBEDDING

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        # Ошибку не глотаем и нулевой вектор не подставляем: в nano_vectordb
        # нулевой запрос даёт NaN-скоры и случайные top-k чанки, а нулевой
        # чанк — NaN-строку, которая до переиндексации попадает в топ-1 (#283).
        # Повторы при сбое делает сам LightRAG.
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.embedding_dim,
                ),
            )
        except Exception as e:
            logger.error(f"Error in GeminiEmbedding: {e}")
            raise

        embeddings = [emb.values for emb in result.embeddings or []]
        if len(embeddings) != len(texts) or any(not values for values in embeddings):
            raise RuntimeError(
                "GeminiEmbedding: модель вернула не все эмбеддинги "
                f"({len(embeddings)} на {len(texts)} текстов)"
            )
        return [list(values) for values in embeddings if values]
