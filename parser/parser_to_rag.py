"""Модуль для парсинга факультетов НГУ и сохранения данных в ChromaDB."""

import logging

from parser.config.load_config import get_parser_config
from parser.nsu_parser import parse_nsu_faculty
from rag.loader import add_texts

logger = logging.getLogger(__name__)


def parse_and_save_to_vectorstore():
    """Парсит все факультеты из конфигурации и сохраняет данные в ChromaDB."""
    logger.info("Начало парсинга факультетов НГУ...")

    config = get_parser_config()
    faculties = config.get("faculties", {})

    if not faculties:
        logger.warning("Список факультетов в конфигурации пуст!")
        return

    all_texts = []

    for fac_code, fac_path in faculties.items():
        logger.info(f"Обработка факультета: {fac_code} ({fac_path})")

        try:
            data = parse_nsu_faculty(f"{fac_path}/")

            if not data:
                logger.warning(f"Не удалось спарсить факультет {fac_code}")
                continue

            # Собираем части страницы в один список
            page_parts = []

            # Добавляем заголовок и описание
            title = data.get("title", "")
            if title:
                page_parts.append(f"Факультет: {title}")

            description = data.get("description", "")
            if description:
                page_parts.append(f"Описание: {description}")

            keywords = data.get("keywords", "")
            if keywords:
                page_parts.append(f"Ключевые слова: {keywords}")

            # Добавляем блоки контента
            content_blocks = data.get("content_blocks", [])
            if content_blocks:
                page_parts.extend(content_blocks)

            # Объединяем всё в одну строку (один документ)
            full_page_text = "\n\n".join(page_parts)

            if full_page_text:
                all_texts.append(full_page_text)
                logger.info(
                    f"  Факультет: {title}, сформирован единый документ длиной {len(full_page_text)} символов"  # noqa: E501
                )

        except Exception as e:
            logger.error(f"Ошибка при обработке факультета {fac_code}: {e}")
            continue

    if all_texts:
        logger.info(f"Сохранение {len(all_texts)} документов (страниц) в ChromaDB...")
        try:
            add_texts(all_texts)
            logger.info("✓ Данные успешно сохранены в векторную базу данных")
        except Exception as e:
            logger.error(f"Ошибка при сохранении в ChromaDB: {e}")
    else:
        logger.warning("Нет данных для сохранения в ChromaDB")

    logger.info("Парсинг завершен")
