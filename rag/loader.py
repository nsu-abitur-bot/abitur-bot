import asyncio
import logging

from rag.graph_memory import get_graph_memory

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_ID = "abitur_kb"


def add_texts(
    texts: list[str],
    source_ids: list[str] | None = None,
    file_paths: list[str] | None = None,
):
    """Добавляет тексты в графовую память.

    Args:
        texts: Список текстов для добавления
        source_ids: URL-идентификаторы документов (для дедупликации)
        file_paths: Строки со всеми URL источниками каждого документа (для цитирования)
    """
    if not texts:
        logger.warning("Список текстов пуст, нечего добавлять")
        return

    logger.info(f"Добавление {len(texts)} текстов в графовую память...")

    graph_memory = get_graph_memory()

    async def _save_all():
        try:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                source_id = (
                    source_ids[i] if source_ids and i < len(source_ids) else None
                )
                file_path = (
                    file_paths[i] if file_paths and i < len(file_paths) else source_id
                )
                # LightRAG handles chunking internally
                await graph_memory.save(
                    DEFAULT_GRAPH_ID,
                    text,
                    source_id=source_id,
                    file_paths_str=file_path,
                )
        finally:
            # Ensure storages are finalized to persist data
            await graph_memory.cleanup(DEFAULT_GRAPH_ID)

    try:
        asyncio.run(_save_all())
        logger.info("Тексты успешно добавлены в графовую память")
    except Exception as e:
        logger.error(f"Ошибка добавления текстов в графовую память: {e}")
