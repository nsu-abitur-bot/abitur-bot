"""Модуль для парсинга страниц и сохранения данных в RAG."""

import logging

from abbrev.expander import get_abbrev_expander
from api.services.document_indexer import index_document
from api.services.document_update import calculate_content_hash, fetch_url_bytes
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.document import DocumentService
from parser.url import process_url
from rag.loader import DEFAULT_GRAPH_ID

logger = logging.getLogger(__name__)


async def parse_and_save_url(url: str, title: str) -> bool:
    """Парсит документ или страницу по URL и сохраняет в векторную базу."""
    logger.info(f"Начало парсинга URL: {url}")
    text = await process_url(url)

    if not text:
        logger.error(f"Не удалось получить контент по URL: {url}")
        return False

    prepared_text = get_abbrev_expander().expand(text.strip())

    logger.info(
        f"Получен текст длиной {len(prepared_text)} символов. Сохранение в RAG..."
    )

    # В качестве source_id используем заголовок или сам URL, если заголовка нет
    source_id = title if title else url

    try:
        try:
            raw = await fetch_url_bytes(url)
            content_hash = calculate_content_hash(raw)
        except Exception:
            content_hash = calculate_content_hash(prepared_text.encode("utf-8"))
        async with AsyncSessionLocal() as session:
            document_service = DocumentService(session)
            document = await document_service.create_or_update_for_source(
                graph_id=DEFAULT_GRAPH_ID,
                title=source_id,
                source_url=url,
            )
            indexed_doc = await index_document(
                document=document,
                text=prepared_text,
                content_hash=content_hash,
                file_path=url,
                graph_id=DEFAULT_GRAPH_ID,
                session=session,
                document_service=document_service,
            )

        if indexed_doc:
            logger.info(f"Успешно сохранён документ {url} в RAG")
            return True
        else:
            logger.warning(f"Документ {url} не был сохранён в RAG")
            return False
    except Exception as e:
        logger.error(f"Ошибка при сохранении документа в RAG ({url}): {e}")
        return False
