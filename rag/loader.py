import logging

from chonkie.chunker import RecursiveChunker

from rag.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def add_texts(texts: list[str]):
    """Добавляет тексты в векторную базу данных.

    Args:
        texts: Список текстов для добавления
    """
    if not texts:
        logger.warning("Список текстов пуст, нечего добавлять")
        return

    chunker = RecursiveChunker(chunk_size=500)

    all_chunks: list[str] = []
    for text in texts:
        if not text or not text.strip():
            continue
        chunks = chunker.chunk(text)
        for chunk in chunks:
            all_chunks.append(chunk.text)

    if not all_chunks:
        logger.warning("После чанкинга не получилось чанков")
        return

    logger.info(f"Добавление {len(all_chunks)} чанков в векторную базу...")
    db = get_vectorstore()
    db.add_texts(all_chunks)
