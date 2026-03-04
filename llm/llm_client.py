import logging
import re
from contextlib import suppress
from html import escape
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from db.redis_client import RedisClient
from faq.faq_matcher import get_faq_matcher
from llm.factory import get_llm_provider
from rag.retriever import query_graph

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """
Ты дружелюбный помощник для абитуриентов НГУ (Новосибирский государственный университет).
Твоя основная задача — отвечать на вопросы об университете и поступлении.

Правила:
- Если пользователь здоровается или пишет нейтральную фразу — ответь коротко и дружелюбно,
  затем предложи задать вопрос об НГУ.
- Если вопрос не связан с НГУ — вежливо скажи, что специализируешься только на НГУ.
- Отвечай коротко, без лишних слов.

Форматируй ответ в HTML для Telegram.
Разрешены только теги: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">.
Не используй Markdown.

Используй следующую информацию из базы знаний для ответа на вопрос пользователя:

{context}

Если информации недостаточно, ответь по общим знаниям о НГУ
и честно укажи, что данных мало.
"""

LIGHTRAG_FORMAT_HINT = (
    "Верни ответ в Telegram-совместимом HTML без Markdown. "
    'Разрешены теги <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">.'
)

# Создаем глобальный экземпляр для переиспользования соединения
_redis_client: Optional[RedisClient] = None


def _sanitize_telegram_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?strong>",
        lambda m: "</b>" if m.group(0).startswith("</") else "<b>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"</?em>",
        lambda m: "</i>" if m.group(0).startswith("</") else "<i>",
        text,
        flags=re.IGNORECASE,
    )

    allowed_simple = {"b", "i", "u", "s", "code", "pre"}

    def replace_tag(match: re.Match[str]) -> str:
        raw_tag = match.group(0)
        tag = raw_tag.strip("<>").strip()
        is_closing = tag.startswith("/")
        tag_body = tag[1:].strip() if is_closing else tag
        tag_name = tag_body.split()[0].lower() if tag_body else ""

        if tag_name in allowed_simple:
            return f"</{tag_name}>" if is_closing else f"<{tag_name}>"

        if tag_name == "a":
            if is_closing:
                return "</a>"
            href_match = re.search(
                r"href\s*=\s*[\"\']([^\"\']+)[\"\']",
                tag_body,
                flags=re.IGNORECASE,
            )
            if href_match:
                href = escape(href_match.group(1), quote=True)
                return f'<a href="{href}">'
            return ""

        return ""

    text = re.sub(r"<[^>]+>", replace_tag, text)
    return text.strip()


async def get_redis_client() -> RedisClient:
    """Получает или создает Redis клиент."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def ask_local_llm(message: str, session_id: str) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки,
    используется для получения и сохранения истории переписки (контекста)
    между пользователем и ассистентом
    """

    try:
        redis_client = await get_redis_client()

        # 1. Сначала сохраняем сообщение пользователя в историю
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        # 2. Проверяем FAQ — если есть заготовленный ответ, возвращаем сразу
        try:
            faq_matcher = get_faq_matcher()
            faq_answer = faq_matcher.match(message)
            if faq_answer:
                logger.info("FAQ match found, skipping LLM call")
                await redis_client.add_message(
                    session_id, {"role": "assistant", "content": faq_answer}
                )
                return faq_answer
        except Exception as e:
            logger.warning(f"FAQ matcher error: {e}")

        # 3. Получаем контекст из LightRAG
        try:
            rag_query = f"{message}\n\n{LIGHTRAG_FORMAT_HINT}"
            rag_context = await query_graph(rag_query)
            if not rag_context or rag_context.startswith("Error executing query"):
                rag_context = "Релевантный контекст из базы знаний не найден."
        except Exception as e:
            logger.warning(f"LightRAG query error: {e}")
            rag_context = "База знаний временно недоступна."

        # 4. Формируем сообщения и отправляем в LLM провайдер (Cerebras по умолчанию)
        try:
            system_prompt = SYSTEM_PROMPT_BASE.format(context=rag_context)
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Добавляем историю переписки
            history = await redis_client.get_history(session_id)
            for entry in history:
                role = entry.get("role", "")
                entry_content = entry.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=entry_content))
                elif role == "assistant":
                    messages.append(AIMessage(content=entry_content))

            provider = get_llm_provider()
            content = await provider.generate(messages)

            # Удаляем блоки <think>...</think>
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = _sanitize_telegram_html(content)

            if not content:
                content = "Ответ не найден"
        except Exception as e:
            logger.warning(f"Provider generation error: {e}")
            content = "LLM временно недоступна."

        # Сохраняем ответ бота в историю
        await redis_client.add_message(
            session_id, {"role": "assistant", "content": content}
        )

        return content

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Что-то пошло не так"


async def cleanup_redis():
    """Закрывает Redis соединение при завершении работы."""
    global _redis_client
    if _redis_client is not None:
        with suppress(Exception):
            await _redis_client.close()
        _redis_client = None
