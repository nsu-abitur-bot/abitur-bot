import asyncio
import logging
from dataclasses import replace
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from config.instance import get_setting
from parser.pdf import PdfPage, pdf_to_base64_images, read_pdf_pages
from parser.safe_fetch import fetch_url_async
from parser.vision import parse_images_with_llm

logger = logging.getLogger(__name__)


async def process_url(url: str) -> Optional[str]:
    """Скачивает и парсит контент (PDF или HTML) по URL."""

    try:
        response = await fetch_url_async(url)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        logger.info(f"Обработка URL: {url} (Content-Type: {content_type})")

        path = urlparse(url).path.lower()

        if "application/pdf" in content_type or path.endswith(".pdf"):
            logger.info("Обнаружен PDF-документ. Запуск PDF-парсера...")
            return await process_pdf_bytes(response.content)

        # Если это не PDF, возвращаем сырой текст страницы
        # без дополнительного препроцессинга,
        # чтобы его можно было отправить в LLM-cleaner, если нужно
        logger.info("Обнаружена HTML-страница. Запуск базового парсера...")
        return process_html(response.text)

    except Exception as e:
        logger.error(f"Ошибка при загрузке или парсинге {url}: {e}")
        return None


async def process_pdf_bytes(pdf_bytes: bytes) -> str:
    """Преобразует байты PDF в изображения и отправляет в LLM."""
    images = pdf_to_base64_images(pdf_bytes)
    if not images:
        logger.warning("Не удалось извлечь изображения из PDF")
        return ""

    markdown_text = await parse_images_with_llm(images)
    return markdown_text


def process_html(html_text: str) -> str:
    """Извлекает текст из HTML (очень базовый парсинг)."""
    soup = BeautifulSoup(html_text, "html.parser")
    # Удаляем скрипты и стили
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Убираем лишние пустые строки
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n\n".join(lines)


async def extract_pdf_pages(pdf_bytes: bytes) -> list[PdfPage]:
    """Keep physical PDF page numbers through native extraction and OCR."""
    limit = int(get_setting("MAX_DOCUMENT_PAGES", "50"))
    native_pages = await asyncio.to_thread(read_pdf_pages, pdf_bytes, limit)
    return [
        replace(page, text=(await parse_images_with_llm([image])).strip())
        if image
        else page
        for page, image in native_pages
    ]


async def fetch_pdf_pages(url: str) -> list[PdfPage]:
    """Download a public PDF with the same URL checks and limits as RAG import."""
    response = await fetch_url_async(url)
    response.raise_for_status()
    return await extract_pdf_pages(response.content)
