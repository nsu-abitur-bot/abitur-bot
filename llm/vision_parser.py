import base64
import logging
import os
from typing import Any, List, Optional

import httpx
from google import genai
from google.genai import types
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

PARSE_PROMPT = (
    "Твоя задача — перевести содержимое этих отсканированных страниц документа в чистый структурированный Markdown. "
    "Если в документе есть таблицы, НЕ используй табличный формат Markdown. Вместо этого преобразуй "
    "каждую строку таблицы в связный текстовый абзац (например: 'Объект: X, Характеристика: Y'). "
    "Это необходимо для лучшего векторного поиска. Убедись, что связь между заголовками столбцов и значениями ячеек "
    "чётко отражена в тексте. Не добавляй лишних комментариев, только результат."
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


async def parse_images_with_llm(
    images_base64: List[str], provider: Optional[str] = None
) -> str:
    """
    Отправляет base64 изображения страниц в LLM (OpenAI или Gemini) для извлечения текста и таблиц в Markdown.
    Если provider не указан, берет из конфигурации окружения PDF_PARSER_PROVIDER, по умолчанию "gemini".
    """
    if not provider:
        provider = os.getenv("PDF_PARSER_PROVIDER", "openai").lower()

    if not images_base64:
        return ""

    logger.info(
        f"LLM получила {len(images_base64)} изображений. Используем провайдер: {provider}"
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


async def _process_batch_openai(images_b64: List[str]) -> str:
    """Обрабатывает пачку страниц с помощью OpenAI Vision"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY не установлен для разбора PDF")
        return ""

    proxy_url = os.getenv("OPENAI_SOCKS5_PROXY")
    http_client = httpx.AsyncClient(proxy=proxy_url) if proxy_url else None

    client = AsyncOpenAI(api_key=api_key, http_client=http_client)

    model = os.getenv("OPENAI_MODEL_VISION", "gpt-5.4-mini")

    content: List[Any] = [
        {
            "type": "input_text",
            "text": PARSE_PROMPT,
        }
    ]

    for img in images_b64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img}",
                "detail": "high",
            }
        )

    try:
        response = await client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            temperature=0.1,  # Низкая креативность для точного извлечения фактов
            max_output_tokens=8192,  # Достаточный запас для перевода 5 страниц текста
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
    model = os.getenv("GEMINI_MODEL_VISION", "gemini-2.5-flash")

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
