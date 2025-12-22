import asyncio
import logging

from rag.graph_memory import get_graph_memory

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_ID = "abitur_kb"


def add_texts(texts: list[str]):
    """Добавляет тексты в графовую память.

    Args:
        texts: Список текстов для добавления
    """
    if not texts:
        logger.warning("Список текстов пуст, нечего добавлять")
        return

    logger.info(f"Добавление {len(texts)} текстов в графовую память...")

    graph_memory = get_graph_memory()

    async def _save_all():
        try:
            for text in texts:
                if not text or not text.strip():
                    continue
                # LightRAG handles chunking internally
                await graph_memory.save(DEFAULT_GRAPH_ID, text)
        finally:
            # Ensure storages are finalized to persist data
            await graph_memory.cleanup(DEFAULT_GRAPH_ID)

    try:
        asyncio.run(_save_all())
        logger.info("Тексты успешно добавлены в графовую память")
    except Exception as e:
        logger.error(f"Ошибка добавления текстов в графовую память: {e}")
