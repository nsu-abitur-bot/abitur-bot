"""
Модуль для чтения markdown файлов из папки baza и сохранения данных в графовую память.
"""

import logging
import shutil
from pathlib import Path

from rag.loader import add_texts

logger = logging.getLogger(__name__)


def clear_data_directory():
    """Удаляет папку data/lightrag перед новым построением графа."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "lightrag"
    if data_dir.exists():
        logger.info(f"Удаление старой базы знаний: {data_dir}")
        shutil.rmtree(data_dir)
    else:
        logger.info("Папка data/lightrag не найдена, пропускаем удаление.")


def _extract_sources(text: str, fallback: str) -> tuple[str, str]:
    """Ищет 'Источники: <url1>, <url2>' или 'Источник: <url>' в начале файла.

    Возвращает (primary_url, all_urls_joined) для использования в ids и file_paths.
    """
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("источники:"):
            urls_str = stripped.split(":", 1)[1].strip()
            urls = [u.strip() for u in urls_str.split(",") if u.strip()]
            if urls:
                return urls[0], ", ".join(urls)
        elif lower.startswith("источник:"):
            url = stripped.split(":", 1)[1].strip()
            if url:
                return url, url
    return fallback, fallback


def parse_baza_and_save_to_memory():
    """Читает все .md файлы из папки baza и сохраняет данные в графовую память."""
    clear_data_directory()
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
    all_source_ids = []
    all_file_paths = []

    for md_file in md_files:
        logger.info(f"Обработка файла: {md_file.name}")

        try:
            text = md_file.read_text(encoding="utf-8")

            if not text.strip():
                logger.warning(f"Файл {md_file.name} пуст")
                continue

            source_id, file_paths_str = _extract_sources(text, fallback=md_file.name)
            all_texts.append(text)
            all_source_ids.append(source_id)
            all_file_paths.append(file_paths_str)
            logger.info(
                f"  Файл: {md_file.name}, источник: {source_id}, "
                f"прочитано {len(text)} символов"
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {md_file.name}: {e}")
            continue

    if all_texts:
        logger.info(f"Сохранение {len(all_texts)} документов в графовую память...")
        try:
            add_texts(all_texts, source_ids=all_source_ids, file_paths=all_file_paths)
            logger.info("✓ Данные из baza успешно сохранены в графовую память")
        except Exception as e:
            logger.error(f"Ошибка при сохранении в графовую память: {e}")
    else:
        logger.warning("Нет данных из baza для сохранения")

    logger.info("Обработка baza завершена")


if __name__ == "__main__":
    parse_baza_and_save_to_memory()
