import logging
import re
from contextlib import suppress
from html import escape
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message import MessageService
from db.redis_client import RedisClient
from faq.faq_matcher import get_faq_matcher
from llm.factory import get_llm_provider
from rag.retriever import query_graph_with_sources

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

{sources_hint}"""

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


async def _save_message_to_pg(
    user_id: int, session_id: str, user_text: str, bot_response: str
) -> None:
    """Сохраняет пару вопрос/ответ в PostgreSQL."""
    try:
        async with AsyncSessionLocal() as db_session:
            service = MessageService(db_session)
            await service.create_message(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                bot_response=bot_response,
            )
    except Exception as e:
        logger.error(f"Ошибка сохранения сообщения в PostgreSQL: {e}")


async def ask_local_llm(message: str, session_id: str, user_id: int = 0) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки,
    используется для получения и сохранения истории переписки (контекста)
    между пользователем и ассистентом
    user_id - идентификатор пользователя Telegram для сохранения в БД
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
                if user_id:
                    await _save_message_to_pg(user_id, session_id, message, faq_answer)
                return faq_answer
        except Exception as e:
            logger.warning(f"FAQ matcher error: {e}")

        # 3. Получаем контекст из LightRAG
        rag_sources: list[str] = []
        try:
            rag_query = f"{message}\n\n{LIGHTRAG_FORMAT_HINT}"
            rag_context, rag_sources = await query_graph_with_sources(rag_query)
            if not rag_context or rag_context.startswith("Error executing query"):
                rag_context = "Релевантный контекст из базы знаний не найден."
                rag_sources = []
            else:
                # Убираем все виды ссылок, которые LightRAG вставляет в ответ,
                # чтобы LLM использовала только те URL, что мы передадим явно.
                # 1. Блок «Источники: ...» до конца текста
                rag_context = re.sub(
                    r"\n{0,2}Источники?:[\s\S]*$",
                    "",
                    rag_context,
                    flags=re.IGNORECASE,
                )
                # 2. «Источник информации (https://...)"
                rag_context = re.sub(
                    r"\s*Источник\s+информации\s*\([^)]*\)",
                    "",
                    rag_context,
                    flags=re.IGNORECASE,
                )
                # 3. Нумерованные ссылки «[N] https://..."
                rag_context = re.sub(
                    r"\n*\[\d+\]\s+https?://\S+",
                    "",
                    rag_context,
                )
                rag_context = rag_context.strip()
        except Exception as e:
            logger.warning(f"LightRAG query error: {e}")
            rag_context = "База знаний временно недоступна."

        # 4. Формируем сообщения и отправляем в LLM провайдер (Cerebras по умолчанию)
        try:
            if rag_sources:
                safe_url = escape(rag_sources[0])
                sources_hint = (
                    f'Если уместно, упомяни в конце ответа официальный сайт: '
                    f'<a href="{safe_url}">{safe_url}</a>'
                )
            else:
                sources_hint = ""
            system_prompt = SYSTEM_PROMPT_BASE.format(context=rag_context, sources_hint=sources_hint)
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

        # Сохраняем в PostgreSQL
        if user_id:
            await _save_message_to_pg(user_id, session_id, message, content)

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
