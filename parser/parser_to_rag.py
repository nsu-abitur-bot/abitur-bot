"""Модуль для парсинга страниц и сохранения данных в RAG."""

import logging

from parser.url import process_url
from rag.loader import add_texts_async

logger = logging.getLogger(__name__)


async def parse_and_save_url(url: str, title: str) -> bool:
    """Парсит документ или страницу по URL и сохраняет в векторную базу."""
    logger.info(f"Начало парсинга URL: {url}")
    text = await process_url(url)

    if not text:
        logger.error(f"Не удалось получить контент по URL: {url}")
        return False

    logger.info(f"Получен текст длиной {len(text)} символов. Сохранение в RAG...")

    # В качестве source_id используем заголовок или сам URL, если заголовка нет
    source_id = title if title else url

    try:
        saved_count = await add_texts_async(
            texts=[text], source_ids=[source_id], file_paths=[url]
        )
        if saved_count > 0:
            logger.info(f"Успешно сохранён документ {url} в RAG")
            return True
        else:
            logger.warning(f"Документ {url} не был сохранён в RAG")
            return False
    except Exception as e:
        logger.error(f"Ошибка при сохранении документа в RAG ({url}): {e}")
        return False


