import asyncio
import logging
import threading

from rag.graph_memory import get_graph_memory

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_ID = "abitur_kb"


async def add_texts_async(
    texts: list[str],
    graph_id: str = DEFAULT_GRAPH_ID,
    source_ids: list[str] | None = None,
    file_paths: list[str] | None = None,
) -> int:
    """Асинхронно добавляет тексты в графовую память.

    Args:
        texts: Список текстов для добавления.
        graph_id: ID графа для сохранения данных.
        source_ids: Названия документов (id для дедупликации/обновления).
        file_paths: URL источники каждого документа.

    Returns:
        Количество успешно сохраненных текстов.
    """
    if not texts:
        logger.warning("Список текстов пуст, нечего добавлять")
        return 0

    logger.info(f"Добавление {len(texts)} текстов в графовую память...")

    graph_memory = get_graph_memory()
    saved_count = 0

    try:
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            source_id = source_ids[i] if source_ids and i < len(source_ids) else None
            file_path = (
                file_paths[i] if file_paths and i < len(file_paths) else source_id
            )

            # Формируем строку без запятых, чтобы избежать проблем со split(",")
            # в query_with_sources
            safe_url = file_path.replace(",", "%2C") if file_path else "Нет"

            # LightRAG handles chunking internally
            saved = await graph_memory.save(
                graph_id,
                text,
                source_id=source_id,  # Теперь здесь будет title
                file_paths_str=safe_url,  # Здесь только URL
            )
            if saved:
                saved_count += 1
    finally:
        # Ensure storages are finalized to persist data
        await graph_memory.cleanup(graph_id)
        
        if saved_count > 0:
            # Очищаем кеш LLM LightRAG, чтобы ответы "Не знаю" не зависали в кеше
            await graph_memory.clear_cache(graph_id)

    logger.info(
        f"Тексты добавлены в графовую память: успешно {saved_count} из {len(texts)}"
    )
    return saved_count


def add_texts(
    texts: list[str],
    source_ids: list[str] | None = None,
    file_paths: list[str] | None = None,
):
    """Добавляет тексты в графовую память.

    Args:
        texts: Список текстов для добавления
        source_ids: Названия документов (id для дедупликации/обновления).
        file_paths: URL источники каждого документа.
    """
    coro = add_texts_async(
        texts,
        graph_id=DEFAULT_GRAPH_ID,
        source_ids=source_ids,
        file_paths=file_paths,
    )

    try:
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            error: list[BaseException] = []

            def _runner() -> None:
                try:
                    asyncio.run(coro)
                except BaseException as exc:  # noqa: BLE001
                    error.append(exc)

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error[0]
        else:
            asyncio.run(coro)

        logger.info("Тексты успешно добавлены в графовую память")
    except Exception as e:
        logger.error(f"Ошибка добавления текстов в графовую память: {e}")
