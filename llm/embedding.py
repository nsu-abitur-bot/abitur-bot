"""One embedding contract for FAQ, staging and live graph callbacks."""

import asyncio

import numpy as np

from config.instance import get_setting, snapshot_values
from llm.factory import get_faq_provider


class UnifiedEmbedding:
    def __init__(self) -> None:
        self.embedding_dim = int((snapshot_values() or {}).get("_embedding_dim", "0"))
        if self.embedding_dim < 1:
            raise ValueError("Размерность модели эмбеддингов ещё не проверена")
        self.model = get_setting("EMBEDDING_MODEL", "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = get_faq_provider().get_embeddings_model()
        if client is None:
            raise ValueError("Модель эмбеддингов недоступна")
        vectors = client.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("Получен неполный набор эмбеддингов")
        for vector in vectors:
            array = np.asarray(vector, dtype=np.float32)
            if (
                array.shape != (self.embedding_dim,)
                or not np.isfinite(array).all()
                or np.linalg.norm(array) <= 0
            ):
                raise ValueError("Получен несовместимый вектор")
        return vectors

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aclose(self) -> None:
        pass
