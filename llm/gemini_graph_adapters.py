import logging
from typing import List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_EMBEDDING_DIMS: dict[str, int] = {
    "gemini-embedding-2-preview": 768,
    "text-embedding-004": 768,
}


class GeminiLLM:
    """LLM-адаптер для LightRAG на базе Google Gemini."""

    def __init__(
        self, api_key: str, model: str = "gemini-3.1-flash-lite-preview"
    ) -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            # If there's a system prompt, we might need to handle it.
            # In google-genai, system instructions can be passed via config.
            system_prompt = kwargs.get("system_prompt")
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.2),
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

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.embedding_dim,
                ),
            )
            embeddings: List[List[float]] = []
            if result.embeddings:
                for emb in result.embeddings:
                    # В некоторых случаях values может быть None,
                    # дадим подсказку анализатору
                    val = (
                        emb.values
                        if emb.values is not None
                        else [0.0] * self.embedding_dim
                    )
                    embeddings.append(val)
            else:
                # Fallback if somehow empty
                embeddings = [[0.0] * self.embedding_dim for _ in texts]
            return embeddings
        except Exception as e:
            logger.error(f"Error in GeminiEmbedding: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]
