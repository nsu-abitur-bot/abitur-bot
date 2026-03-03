"""
Модуль для чтения markdown файлов из папки baza и сохранения данных в графовую память.
"""

import logging
from pathlib import Path

from rag.loader import add_texts

logger = logging.getLogger(__name__)


def parse_baza_and_save_to_memory():
    """Читает все .md файлы из папки baza и сохраняет данные в графовую память."""
    logger.info("Начало обработки файлов из папки baza...")

    # Определяем путь к папке baza относительно корня проекта
    # Предполагаем, что скрипт находится в parser/
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    baza_dir = project_root / "baza"

    if not baza_dir.exists():
        logger.error(f"Папка {baza_dir} не найдена!")
        return

    md_files = list(baza_dir.glob("*.md"))

    if not md_files:
        logger.warning(f"В папке {baza_dir} не найдено markdown файлов")
        return

    all_texts = []

    for md_file in md_files:
        logger.info(f"Обработка файла: {md_file.name}")

        try:
            text = md_file.read_text(encoding="utf-8")

            if not text.strip():
                logger.warning(f"Файл {md_file.name} пуст")
                continue

            # Добавляем содержимое файла как есть.
            all_texts.append(text)
            logger.info(f"  Файл: {md_file.name}, прочитано {len(text)} символов")

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {md_file.name}: {e}")
            continue

    if all_texts:
        logger.info(f"Сохранение {len(all_texts)} документов в графовую память...")
        try:
            add_texts(all_texts)
            logger.info("✓ Данные из baza успешно сохранены в графовую память")
        except Exception as e:
            logger.error(f"Ошибка при сохранении в графовую память: {e}")
    else:
        logger.warning("Нет данных из baza для сохранения")

    logger.info("Обработка baza завершена")
