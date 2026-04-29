import logging
import re
from contextlib import suppress
from html import escape
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message import MessageService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.user import UserService
from db.redis_client import RedisClient
from bot.utils import normalize_url_for_messaging
from faq.faq_matcher import get_faq_matcher
from llm.factory import get_llm_provider
from llm.profiles import LLMProfiles
from rag.retriever import query_graph_with_sources

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_BASE = """
Ты — официальный дружелюбный помощник-бот для абитуриентов НГУ
(Новосибирский государственный университет).

Правила поведения:
1. Имя и легкий диалог: Если пользователь здоровается или говорит о себе (например,
   называет своё имя), обязательно используй историю переписки, чтобы поддержать
   беседу и обратиться по имени.
2. Вопросы об НГУ: Ищи фактическую информацию ИСКЛЮЧИТЕЛЬНО в блоке
   "Контекст из базы знаний об НГУ" ниже. Если информации о предмете вопроса
   (факультет, программа, цифры и любые другие детали) нет в твоем контексте —
   ОБЯЗАТЕЛЬНО ответь: "Я не нашел информации об этом в базе знаний НГУ".
   Категорически запрещено давать общие советы, запрещено давать ссылки
   (если их нет в переданном контексте), и запрещено отвечать,
   используя свои собственные "обученные" общие знания об НГУ.
3. Оффтоп: Если вопрос вообще не про НГУ и не является поддержанием диалога,
   вежливо скажи, что ты консультируешь только по вопросам НГУ.
4. Отвечай коротко, без лишней воды. Структурируй абзацы.

Форматирование в HTML (для Telegram):
Разрешены только теги: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">.
Не используй Markdown (**жирный** или *курсив*). Оборачивай жирный шрифт в <b>.

Контекст из базы знаний об НГУ:
{context}

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
                raw_href = href_match.group(1)
                normalized_href = normalize_url_for_messaging(raw_href)
                href = escape(normalized_href, quote=True)
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
            user_service = UserService(db_session)
            await user_service.ensure_user_exists(user_id)

            message_service = MessageService(db_session)
            await message_service.create_message(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                bot_response=bot_response,
            )
    except Exception as e:
        logger.error(f"Ошибка сохранения сообщения в PostgreSQL: {e}")


async def _save_log_to_db(
    user_id: int,
    session_id: str,
    message_type: str,
    content: str,
    message_metadata: dict,
) -> None:
    """Сохраняет лог в таблицу message_logs."""
    try:
        async with AsyncSessionLocal() as db_session:
            log_service = MessageLogService(db_session)
            await log_service.create_log(
                user_id=user_id,
                session_id=session_id,
                message_type=message_type,
                content=content,
                message_metadata=message_metadata,
            )
    except Exception as e:
        logger.error(f"Ошибка сохранения лога в БД: {e}")


async def ask_local_llm(message: str, session_id: str, user_id: int = 0) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки,
    используется для получения и сохранения истории переписки (контекста)
    между пользователем и ассистентом
    user_id - внутренний идентификатор пользователя для сохранения в БД
    """
    logger.info(
        f"[{session_id}] New message received from user={user_id}: {message[:50]}..."
    )

    try:
        redis_client = await get_redis_client()

        # 1. Сначала сохраняем сообщение пользователя в историю
        logger.info(f"[{session_id}] Saving user message to Redis history.")
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        # 2. Проверяем FAQ — если есть заготовленный ответ, возвращаем сразу
        logger.info(f"[{session_id}] Checking FAQ for match.")
        try:
            faq_matcher = get_faq_matcher()
            faq_answer = faq_matcher.match(message)
            if faq_answer:
                logger.info(
                    f"[{session_id}] FAQ match found, returning predefined answer."
                )
                # Логируем что решил ответить из FAQ
                logger.info(f"[{session_id}] FAQ result:")
                logger.info(f"[{session_id}] - Matched FAQ answer: {faq_answer}")

                # Сохраняем лог в БД
                await _save_log_to_db(
                    user_id=user_id,
                    session_id=session_id,
                    message_type="faq_match",
                    content=faq_answer,
                    message_metadata={"source": "faq"},
                )
                await redis_client.add_message(
                    session_id, {"role": "assistant", "content": faq_answer}
                )
                if user_id:
                    await _save_message_to_pg(user_id, session_id, message, faq_answer)
                return faq_answer
        except Exception as e:
            logger.warning(f"FAQ matcher error: {e}")

        # 3. Быстрая проверка на необходимость RAG
        # (чтобы экономить вызовы графа для "привет", "как дела" и пр.)
        need_rag = True
        try:
            intent_prompt = (
                "Ты — маршрутизатор. Определи, нужно ли искать информацию "
                "в базе знаний НГУ, чтобы ответить на сообщение пользователя.\n"
                "Ответь ТОЛЬКО 'YES', если вопрос касается НГУ, учебы, "
                "поступления, общежитий, или любых фактов.\n"
                "Ответь ТОЛЬКО 'NO', если это простое приветствие, разговор "
                "на отвлеченные темы (chit-chat), "
                "вопрос о собеседнике, или благодарность.\n"
                "Не пиши ничего кроме 'YES' или 'NO'."
            )
            intent_messages: list[BaseMessage] = [
                SystemMessage(content=intent_prompt),
                HumanMessage(content=message),
            ]
            intent_provider = get_llm_provider()
            intent_response = await intent_provider.generate(intent_messages, profile=LLMProfiles.INTENT)
            intent_response = (
                re.sub(r"<think>.*?</think>", "", intent_response, flags=re.DOTALL)
                .strip()
                .upper()
            )
            # Извлекаем только "YES" или "NO" из ответа
            match_yes = re.search(r"\bYES\b", intent_response)
            match_no = re.search(r"\bNO\b", intent_response)

            if match_no and not match_yes:
                need_rag = False
                logger.info(f"Skipping RAG for conversational query: {message}")
            elif not match_yes and not match_no:
                logger.warning(
                    f"Unexpected intent response: {intent_response},"
                    + " falling back to full RAG"
                )

        except Exception as e:
            logger.warning(
                f"Intent classification error: {e}, falling back to full RAG"
            )

        # 4. Получаем контекст из LightRAG (если нужно)
        rag_sources: list[str] = []
        rag_context = ""
        if need_rag:
            logger.info(f"[{session_id}] Querying LightRAG for context.")
            try:
                rag_query = f"{message}\n\n{LIGHTRAG_FORMAT_HINT}"
                rag_context_raw, rag_sources = await query_graph_with_sources(rag_query)
                if not rag_context_raw or rag_context_raw.startswith(
                    "Error executing query"
                ):
                    logger.info(f"[{session_id}] No relevant context found in RAG.")
                    rag_context = "Релевантный контекст из базы знаний не найден."
                    rag_sources = []
                    logger.info(
                        f"[{session_id}] RAG retrieval result: No context found"
                    )
                else:
                    logger.info(
                        f"[{session_id}] Retrieved context from RAG "
                        f"(sources: {len(rag_sources)})."
                    )
                    rag_context = rag_context_raw
                    # Логируем что достали из RAG
                    logger.info(f"[{session_id}] RAG retrieval result:")
                    logger.info(
                        f"[{session_id}] - Context (first 500 chars): "
                        f"{rag_context_raw[:500]}..."
                    )
                    logger.info(
                        f"[{session_id}] - Sources ({len(rag_sources)}): {rag_sources}"
                    )

                    # Сохраняем лог RAG в БД
                    await _save_log_to_db(
                        user_id=user_id,
                        session_id=session_id,
                        message_type="rag_context",
                        content=rag_context_raw[:1000],  # ограничиваем размер
                        message_metadata={
                            "sources": rag_sources,
                            "context_length": len(rag_context_raw),
                            "sources_count": len(rag_sources),
                        },
                    )
                # Убираем все виды ссылок, которые LightRAG вставляет в ответ,
                # чтобы LLM использовала только те URL, что мы передадим явно.
                # 1. Блок «Источники: ...» и аналогичные до конца строки
                rag_context = re.sub(
                    r"(?im)^.*?(?:###\s*)?(?:Источники?|References?|Ссылки):?\s*[^\n]*",
                    "",
                    rag_context,
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

        # 5. Формируем сообщения и отправляем в LLM провайдер (Cerebras по умолчанию)
        try:
            logger.info(f"[{session_id}] Preparing prompt to LLM provider.")
            valid_urls = [
                url
                for url in rag_sources
                if url.startswith("http://") or url.startswith("https://")
            ]
            if valid_urls:
                links_text = "\n".join([f"- {url}" for url in valid_urls[:5]])
                sources_hint = (
                    "\n\nИНСТРУКЦИЯ К ОТВЕТУ:\n"
                    "ЕСЛИ ты использовал информацию из блока 'Контекст из базы знаний"
                    + " об НГУ' для ответа, "
                    "внимательно просмотри список доступных ссылок ниже."
                    + " Выбери из них ТОЛЬКО те, "
                    "которые непосредственно относятся к твоему ответу.\n"
                    "Затем обязательно добавь в самый конец своего ответа"
                    + " выбранные ссылки в следующем формате:\n\n"
                    "<b>Источники:</b>\n"
                    '<a href="URL_1">Название документа или страницы</a>\n'
                    '<a href="URL_2">Понятное описание источника</a>\n\n'
                    "(Но если ты просто здороваешься, говоришь на отвлеченные темы или"
                    + " не нашел ответа в Контексте - блок "
                    + "'Источники:' ВООБЩЕ НЕ добавляй!)\n"
                    "КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ придумывать свои ссылки или брать "
                    "их из истории переписки. "
                    "Используй ТОЛЬКО ссылки из списка ниже.\n\n"
                    "СПИСОК ДОСТУПНЫХ ССЫЛОК ИЗ БАЗЫ ЗНАНИЙ (выбери подходящие):\n"
                    f"{links_text}"
                )
            else:
                sources_hint = ""
            system_prompt = SYSTEM_PROMPT_BASE.format(
                context=rag_context, sources_hint=sources_hint
            )
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Добавляем историю переписки
            history = await redis_client.get_history(session_id)
            for entry in history:
                role = entry.get("role", "")
                entry_content = entry.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=entry_content))
                elif role == "assistant":
                    # Очищаем старые ответы от блока с источниками,
                    # чтобы они не сбивали с толку LLM
                    entry_clean = re.sub(
                        r"(?i)\n?(?:<br>|<b>|###\s*|\*+\s*)*\s*(?:Источники|Источник|References|Ссылки)(?:\s+информации)?:?\s*(?:</b>|\*+)*\s*(?:\n|<a)[\s\S]*",
                        "",
                        entry_content,
                    )
                    # Также удаляем любые оставшиеся HTML-ссылки
                    entry_clean = re.sub(
                        r"<a\s+href=[^>]+>.*?</a>", "", entry_clean
                    ).strip()
                    messages.append(AIMessage(content=entry_clean))

            provider = get_llm_provider()
            logger.info(
                f"[{session_id}] Sending payload to LLM "
                f"({provider.__class__.__name__})."
            )
            content = await provider.generate(messages, profile=LLMProfiles.CHAT)

            # Пост-процессинг: удаляем "левые" ссылки, которых не было в valid_urls
            if valid_urls:
                # Находим все ссылки вида <a href="...">...</a>
                def _filter_links(match):
                    link_html = match.group(0)
                    href_val = re.search(r'href=["\']([^"\']+)["\']', link_html, re.I)
                    if href_val:
                        found_url = href_val.group(1).strip()
                        # Если URL не из нашего списка базы знаний, вырезаем его из ответа
                        if found_url not in valid_urls:
                            return match.group(
                                2
                            )  # возвращаем только текст ссылки без тега
                    return link_html

                content = re.sub(
                    r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                    _filter_links,
                    content,
                    flags=re.IGNORECASE,
                )

            logger.info(f"[{session_id}] Received response from LLM.")

            # Логируем что решил ответить LLM
            logger.info(f"[{session_id}] LLM response result:")
            logger.info(
                f"[{session_id}] - Raw response (first 500 chars): {content[:500]}..."
            )
            logger.info(f"[{session_id}] - Response length: {len(content)} characters")

            # Сохраняем лог LLM ответа в БД
            await _save_log_to_db(
                user_id=user_id,
                session_id=session_id,
                message_type="llm_response",
                content=content[:2000],  # ограничиваем размер
                message_metadata={
                    "response_length": len(content),
                    "provider": provider.__class__.__name__,
                },
            )

            # Удаляем блоки <think>...</think>
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = _sanitize_telegram_html(content)

            if not content:
                logger.warning(
                    f"[{session_id}] LLM returned empty content after sanitization."
                )
                content = "Ответ не найден"
                logger.info(
                    f"[{session_id}] Final LLM response after sanitization: {content}"
                )
            else:
                logger.info(
                    f"[{session_id}] Final LLM response after sanitization "
                    f"(first 300 chars): {content[:300]}..."
                )
        except Exception as e:
            logger.warning(f"[{session_id}] Provider generation error: {e}")
            content = "LLM временно недоступна."
            logger.info(f"[{session_id}] Fallback response due to error: {content}")

        # Сохраняем ответ бота в историю
        logger.info(f"[{session_id}] Saving bot response to Redis history.")
        await redis_client.add_message(
            session_id, {"role": "assistant", "content": content}
        )

        # Сохраняем в PostgreSQL
        if user_id:
            logger.info(f"[{session_id}] Saving message pair to PostgreSQL.")
            await _save_message_to_pg(user_id, session_id, message, content)

        logger.info(f"[{session_id}] Message processing complete.")
        return content

    except Exception as e:
        logger.error(f"[{session_id}] LLM error: {e}")
        return "Что-то пошло не так"


async def cleanup_redis():
    """Закрывает Redis соединение при завершении работы."""
    global _redis_client
    if _redis_client is not None:
        with suppress(Exception):
            await _redis_client.close()
        _redis_client = None
