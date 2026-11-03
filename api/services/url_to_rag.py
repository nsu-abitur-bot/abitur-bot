"""Парсинг страниц в очередь предложений редактору."""

import logging

from abbrev.expander import get_abbrev_expander
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.content_proposal import ContentProposalService
from parser.url import process_url
from rag.loader import DEFAULT_GRAPH_ID

logger = logging.getLogger(__name__)


async def parse_and_save_url(url: str, title: str) -> bool:
    """Парсит документ или страницу по URL и ставит результат на проверку."""
    logger.info(f"Начало парсинга URL: {url}")
    text = await process_url(url)

    if not text:
        logger.error(f"Не удалось получить контент по URL: {url}")
        return False

    prepared_text = get_abbrev_expander().expand(text.strip())

    logger.info("Получен текст длиной %d символов", len(prepared_text))

    # В качестве source_id используем заголовок или сам URL, если заголовка нет
    source_id = title if title else url

    try:
        async with AsyncSessionLocal() as session:
            await ContentProposalService(session).create(
                source="parser",
                graph_id=DEFAULT_GRAPH_ID,
                title=source_id,
                source_url=url,
                text=prepared_text,
            )
        logger.info("Документ %s ожидает проверки редактором", url)
        return True
    except Exception as e:
        logger.error("Не удалось создать предложение для %s: %s", url, e)
        return False
