import logging
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from lightrag.utils import TiktokenTokenizer

from abbrev.expander import get_abbrev_expander
from llm.factory import get_llm_provider
from llm.profiles import LLMProfiles
from parser.prompts import CONTEXT_ENRICHMENT_PROMPT

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cleanup_tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer()

SYSTEM_PROMPT = f"""Ты - эксперт по очистке и структурированию текстовых данных.
Твоя задача — получить «грязный» текст, спаршенный с веб-страницы или PDF-документа,
и вернуть чистый, хорошо структурированный текст для загрузки в RAG.

Правила:
1. Удали весь веб-мусор: элементы навигации, футеры, хедеры, тексты кнопок, баннеры,
нерелевантные ссылки (например, "Вернуться на главную", "Контакты").
2. Сохрани всю смысловую и фактическую информацию. Не сокращай факты, даты, числа,
списки, контакты и важные детали.
3. Структурируй результат: используй заголовки, списки,
абзацы для логичного разбиения текста.
4. Исправь проблемы с кодировкой, переносами слов, случайными пробелами,
слипшимся текстом.
5. Не придумывай и не добавляй информацию от себя,
работай исключительно с исходным текстом.
6. Категорически запрещено использовать Markdown-таблицы и символ вертикальной черты "|".
{CONTEXT_ENRICHMENT_PROMPT}
"""


async def clean_and_structure_text(raw_text: str, max_chunk_tokens: int = 10000) -> str:
    """
    Очищает и структурирует текст через LLM.
    """
    if not raw_text or not raw_text.strip():
        return ""

    expanded_text = get_abbrev_expander().expand(raw_text)
    profile = LLMProfiles.PARSER
    input_budget = min(max_chunk_tokens, (profile.max_tokens or 4000) // 2)
    try:
        input_tokens = len(_cleanup_tokenizer().encode(expanded_text))
    except Exception:
        logger.warning(
            "Не удалось оценить бюджет очистки; сохранён полный исходный текст"
        )
        return expanded_text
    if input_tokens > input_budget:
        # Keeping every fact can require at least as many output tokens as input.
        # Avoid a paid request that will truncate, without increasing token limits.
        logger.info(
            "LLM-очистка пропущена: %s входных токенов, бюджет %s; "
            "сохранён полный исходный текст",
            input_tokens,
            input_budget,
        )
        return expanded_text

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Очисти и структурируй следующий текст:\n\n{expanded_text}"
        ),
    ]

    try:
        llm = get_llm_provider()
        response_text = await llm.generate(messages, profile=profile)
        return get_abbrev_expander().expand(response_text)
    except Exception as e:
        logger.error(f"Ошибка при очистке текста через LLM: {e}")
        # Возвращаем текст после локальной инъекции, чтобы цепочка сохранила
        # расшифровки даже при недоступности LLM-cleaner.
        return expanded_text


async def generate_title_from_text(text: str) -> str:
    """
    Генерирует короткое и понятное название для документа на основе его текста.
    Используется, чтобы не оставлять загруженные файлы (напр., PDF)
    со скучными техническими названиями.
    """
    if not text or not text.strip():
        return "Без названия"

    llm = get_llm_provider()

    # Берем первые 3000 символов, чтобы не тратить весь контекст
    preview_text = text[:3000]

    messages: list[BaseMessage] = [
        SystemMessage(
            content="Ты - нейросеть, которая придумывает"
            "релевантные названия для документов."
            "Проанализируй текст и верни ТОЛЬКО ОДНО короткое емкое название "
            "(до 7 слов), без кавычек, точек на конце и без пояснений."
        ),
        HumanMessage(
            content=f"Сгенерируй название для этого документа:\n\n{preview_text}"
        ),
    ]

    try:
        response_text = await llm.generate(messages, profile=LLMProfiles.TITLE)
        title = response_text.strip("'\" \n\r.")
        return title if title else "Без названия"
    except Exception as e:
        logger.error(f"Ошибка при генерации заголовка через LLM: {e}")
        return "Без названия"
