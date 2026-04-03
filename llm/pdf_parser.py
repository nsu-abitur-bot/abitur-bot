import base64
import logging
import os
from typing import Any, List, Optional

import fitz
import httpx
from google import genai
from google.genai import types
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

PARSE_PROMPT = (
    "Твоя задача — перевести содержимое этих отсканированных страниц документа в чистый структурированный Markdown. "
    "Строго сохраняй структуру таблиц в Markdown-формате. Не добавляй никаких лишних комментариев и вступительных фраз, "
    "только результат."
)


def _clean_markdown(text: str) -> str:
    """Обрезает лишнее форматирование Markdown (например, ответы с блоками ```markdown ... ```)"""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[11:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def parse_pdf_with_llm(pdf_bytes: bytes, provider: Optional[str] = None) -> str:
    """
    Конвертирует PDF в изображения и отправляет в LLM (OpenAI или Gemini) для извлечения текста и таблиц в Markdown.
    Если provider не указан, берет из конфигурации окружения PDF_PARSER_PROVIDER, по умолчанию "gemini".
    """
    if not provider:
        provider = os.getenv("PDF_PARSER_PROVIDER", "openai").lower()

    images_base64 = _pdf_to_base64_images(pdf_bytes)
    if not images_base64:
        return ""

    logger.info(
        f"PDF конвертирован в {len(images_base64)} изображений. Используем провайдер: {provider}"
    )

    # Бьем на батчи по 5 страниц, чтобы не превысить лимиты окна или ошибки TimeOut на больших документах
    batch_size = 5
    all_markdown = []

    for i in range(0, len(images_base64), batch_size):
        batch = images_base64[i : i + batch_size]
        logger.info(
            f"Обработка страниц {i + 1}-{min(i + batch_size, len(images_base64))} из {len(images_base64)}..."
        )

        if provider == "openai":
            result = await _process_batch_openai(batch)
        else:
            result = await _process_batch_gemini(batch)

        if result:
            all_markdown.append(result)

    return "\n\n".join(all_markdown)


def _pdf_to_base64_images(pdf_bytes: bytes) -> List[str]:
    """Конвертирует страницы PDF во множество base64-строк формата JPEG"""
    images = []
    # fitz.open(stream) -> filetype должен быть "pdf"
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # Масштаб 2.0 дает ~150-300 DPI, что достаточно для качественного OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("jpeg")
            images.append(base64.b64encode(img_bytes).decode("utf-8"))
    return images


async def _process_batch_openai(images_b64: List[str]) -> str:
    """Обрабатывает пачку страниц с помощью OpenAI Vision"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY не установлен для разбора PDF")
        return ""

    proxy_url = os.getenv("OPENAI_SOCKS5_PROXY")
    http_client = httpx.AsyncClient(proxy=proxy_url) if proxy_url else None

    client = AsyncOpenAI(api_key=api_key, http_client=http_client)

    # Можно использовать gpt-4o, он быстрее и мощнее
    model = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")

    content: List[Any] = [
        {
            "type": "text",
            "text": PARSE_PROMPT,
        }
    ]

    for img in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "high"},
            }
        )

    try:
        response = await client.responses.create(
            model=model,
            input=content,
            max_output_tokens=4000,
            temperature=0.1,
        )
        text = response.output_text or ""
        return _clean_markdown(text)
    except Exception as e:
        logger.error(f"Ошибка OpenAI парсинга страниц PDF: {e}")
        return ""


async def _process_batch_gemini(images_b64: List[str]) -> str:
    """Обрабатывает пачку страниц с помощью Gemini Vision"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY не установлен для разбора PDF")
        return ""

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL_VISION", "gemini-3.1-flash-lite-preview")

    contents: List[Any] = [PARSE_PROMPT]
    for img in images_b64:
        contents.append(
            types.Part.from_bytes(data=base64.b64decode(img), mime_type="image/jpeg")
        )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = response.text or ""
        return _clean_markdown(text)
    except Exception as e:
        logger.error(f"Ошибка Gemini парсинга страниц PDF: {e}")
        return ""
