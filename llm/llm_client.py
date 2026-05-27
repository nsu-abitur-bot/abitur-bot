import asyncio
import json
import logging
import re
from contextlib import suppress
from html import escape, unescape
from typing import Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from abbrev.expander import get_abbrev_expander
from bot.utils import normalize_url_for_messaging
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message import MessageService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.topic import TopicService
from db.postgres.services.user import UserService
from db.redis.client import RedisClient
from faq.matcher import get_faq_matcher
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

DEFAULT_SOURCE_TITLE = "Источник информации"

# Создаем глобальный экземпляр для переиспользования соединения
_redis_client: Optional[RedisClient] = None

# Реестр фоновых задач (write-операций, intent-классификации и т.п.),
# которые не блокируют возврат ответа пользователю.
# Держим strong-ref, чтобы GC не убил задачу до завершения.
_background_tasks: set[asyncio.Task] = set()


def _spawn_bg(coro) -> None:
    """Запустить корутину как fire-and-forget background-задачу.

    Задача регистрируется в глобальном set, чтобы избежать преждевременной
    очистки сборщиком мусора (см. asyncio.create_task doc warning).
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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


def _clean_source_url(url: str) -> str:
    url = unescape(url).strip().strip("<>\"'")
    url = url.rstrip(".,;:!?")

    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip()

    return url


def _source_title_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.netloc:
        return parsed.netloc
    return DEFAULT_SOURCE_TITLE


def _add_source(
    sources: list[dict[str, str]],
    seen_urls: set[str],
    url: str,
    title: str | None = None,
) -> None:
    clean_url = _clean_source_url(url)
    if not clean_url.startswith(("http://", "https://")) or clean_url in seen_urls:
        return

    clean_title = (title or "").strip()
    if not clean_title or clean_title.startswith(("http://", "https://")):
        clean_title = _source_title_from_url(clean_url)

    seen_urls.add(clean_url)
    sources.append({"url": clean_url, "title": clean_title})


def _extract_sources_from_rag_context(context: str) -> list[dict[str, str]]:
    """Достает ссылки из текста, который вернул LightRAG, без metadata references."""
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for match in re.finditer(
        r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        context,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        _add_source(sources, seen_urls, match.group(1), title)

    for match in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", context):
        _add_source(sources, seen_urls, match.group(2), match.group(1))

    for match in re.finditer(r"(?m)^\s*[-*]?\s*\[(\d+)\]\s+(https?://\S+)", context):
        _add_source(sources, seen_urls, match.group(2), f"Источник {match.group(1)}")

    for match in re.finditer(r"https?://[^\s<>{}\"']+", context):
        _add_source(sources, seen_urls, match.group(0))

    return sources


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


async def _classify_intent_bg(
    expanded_message: str,
    log_entry_id: Optional[int],
    session_id: str,
) -> None:
    """Фоновая классификация topic_id для аналитики.

    Результат используется только для записи в message_logs.update_log_topic,
    на ответ бота не влияет (need_rag всегда True).
    """
    try:
        async with AsyncSessionLocal() as session:
            topic_service = TopicService(session)
            topics = await topic_service.get_all_active_topics()

        topics_list = (
            "\n".join([f"{topic.id}: {topic.label}" for topic in topics])
            if topics
            else "Нет доступных тем."
        )
        valid_topic_ids = {topic.id for topic in topics} if topics else set()

        intent_prompt = (
            "Ты — маршрутизатор. Выбери подходящую тему для сообщения "
            "пользователя.\n"
            "Ответь СТРОГО в формате JSON:\n"
            '{"is_nsu": true, "topic_id": 123}\n'
            "Где 'is_nsu' всегда true.\n\n"
            "Список тем для 'topic_id':\n"
            f"{topics_list}\n\n"
            "Выбери наиболее подходящий 'topic_id', либо null,"
            "если ни одна тема не подходит."
        )
        intent_messages: list[BaseMessage] = [
            SystemMessage(content=intent_prompt),
            HumanMessage(content=expanded_message),
        ]
        intent_provider = get_llm_provider()
        intent_response = await intent_provider.generate(
            intent_messages, profile=LLMProfiles.INTENT
        )

        topic_id = None
        try:
            json_match = re.search(r"(\{.*\})", intent_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(intent_response)
            topic_id = parsed.get("topic_id")
            if topic_id is not None and topic_id not in valid_topic_ids:
                logger.warning(
                    f"[{session_id}] LLM hallucinated topic_id={topic_id}. Ignoring."
                )
                topic_id = None
        except json.JSONDecodeError:
            logger.warning(
                f"[{session_id}] Failed to parse JSON from intent response: "
                f"{intent_response}"
            )
            return

        if topic_id and log_entry_id:
            try:
                async with AsyncSessionLocal() as session:
                    log_service = MessageLogService(session)
                    await log_service.update_log_topic(log_entry_id, topic_id)
            except Exception as e:
                logger.error(
                    f"[{session_id}] Failed to update topic id in log: {e}"
                )
    except Exception as e:
        logger.warning(f"[{session_id}] Background intent classification error: {e}")


def _build_history_text(history_entries: list[dict], limit: int = 6) -> str:
    """Собирает текст последних N сообщений истории для передачи в RAG-запрос."""
    history_lines = []
    for entry in history_entries[-limit:]:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if not content:
            continue
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines).strip()


def _clean_rag_context(rag_context: str) -> str:
    """Убирает из RAG-контекста все виды ссылок, которые LightRAG вставляет,
    чтобы LLM использовала только те URL, что мы передадим явно."""
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
    return rag_context.strip()


async def ask_local_llm(
    message: str, session_id: str, user_id: int = 0, log_entry_id: Optional[int] = None
) -> str:
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

        # 1. Сохраняем сообщение пользователя в Redis-историю.
        logger.info(f"[{session_id}] Saving user message to Redis history.")
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        # 2. Расширяем аббревиатуры для улучшения FAQ-матчинга и RAG-запросов.
        try:
            expanded_message = get_abbrev_expander().expand(message)
        except Exception:
            expanded_message = message

        # 3. Intent-классификация уезжает в фон — она нужна только для
        # аналитического лога (update_log_topic) и не влияет на ответ.
        _spawn_bg(_classify_intent_bg(expanded_message, log_entry_id, session_id))

        # 4. Параллельно стартуем FAQ-матч и чтение истории.
        # FAQ обычно отвечает за 0.3-1с, history — за ~10мс.
        faq_matcher = get_faq_matcher()
        task_faq = asyncio.create_task(faq_matcher.match_async(expanded_message))
        task_history = asyncio.create_task(redis_client.get_history(session_id))

        # Получаем историю (нужна и для RAG, и для финального промпта).
        try:
            history_entries = await task_history
        except Exception as e:
            logger.warning(f"[{session_id}] Redis get_history error: {e}")
            history_entries = []
        history_text = _build_history_text(history_entries)

        # 5. Запускаем LightRAG параллельно с ожиданием FAQ.
        rag_query = f"{expanded_message}\n\n{LIGHTRAG_FORMAT_HINT}"
        task_rag = asyncio.create_task(
            query_graph_with_sources(
                rag_query, conversation_history=history_text or None
            )
        )

        # 6. Ждём FAQ. Если матч найден — отменяем RAG и быстро возвращаем ответ.
        try:
            faq_answer = await task_faq
        except Exception as e:
            logger.warning(f"[{session_id}] FAQ matcher error: {e}")
            faq_answer = None

        if faq_answer:
            logger.info(f"[{session_id}] FAQ match found, returning predefined answer.")
            task_rag.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task_rag

            # Запись лога FAQ + истории + PG — fire-and-forget.
            _spawn_bg(
                _save_log_to_db(
                    user_id=user_id,
                    session_id=session_id,
                    message_type="faq_match",
                    content=faq_answer,
                    message_metadata={"source": "faq"},
                )
            )
            _spawn_bg(
                redis_client.add_message(
                    session_id, {"role": "assistant", "content": faq_answer}
                )
            )
            if user_id:
                _spawn_bg(
                    _save_message_to_pg(user_id, session_id, message, faq_answer)
                )
            return faq_answer

        # 7. FAQ промахнулся — ждём RAG.
        rag_sources: list[dict] = []
        rag_context = ""
        try:
            logger.info(f"[{session_id}] Awaiting LightRAG context.")
            rag_context_raw, metadata_sources = await task_rag
            if not rag_context_raw or rag_context_raw.startswith(
                "Error executing query"
            ):
                logger.info(f"[{session_id}] No relevant context found in RAG.")
                rag_context = "Релевантный контекст из базы знаний не найден."
                rag_sources = []
            else:
                rag_sources = metadata_sources
                rag_context = rag_context_raw
                logger.info(
                    f"[{session_id}] Retrieved context from RAG "
                    f"(sources: {len(rag_sources)})."
                )
                logger.info(
                    f"[{session_id}] - Context (first 500 chars): "
                    f"{rag_context_raw[:500]}..."
                )
                logger.info(
                    f"[{session_id}] - Sources ({len(rag_sources)}): {rag_sources}"
                )

                _spawn_bg(
                    _save_log_to_db(
                        user_id=user_id,
                        session_id=session_id,
                        message_type="rag_context",
                        content=rag_context_raw[:1000],
                        message_metadata={
                            "sources": rag_sources,
                            "context_length": len(rag_context_raw),
                            "sources_count": len(rag_sources),
                        },
                    )
                )
            rag_context = _clean_rag_context(rag_context)
        except Exception as e:
            logger.warning(f"[{session_id}] LightRAG query error: {e}")
            rag_context = "База знаний временно недоступна."

        # 8. Формируем промпт и зовём основную LLM-генерацию.
        try:
            logger.info(f"[{session_id}] Preparing prompt to LLM provider.")
            valid_sources = rag_sources

            sources_hint = (
                "\n\nИНСТРУКЦИЯ К ОТВЕТУ:\n"
                "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать блок 'Источники' или перечислять ссылки. "
                "Просто ответь на вопрос пользователя, опираясь на контекст!"
            )

            system_prompt = SYSTEM_PROMPT_BASE.format(
                context=rag_context, sources_hint=sources_hint
            )
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

            # Переиспользуем уже полученную историю (без второго Redis-вызова).
            for entry in history_entries:
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
                    entry_clean = re.sub(
                        r"<a\s+href=[^>]+>.*?</a>", "", entry_clean
                    ).strip()
                    messages.append(AIMessage(content=entry_clean))

            provider = get_llm_provider()
            logger.info(
                f"[{session_id}] Sending payload to LLM ({provider.__class__.__name__})."
            )
            content = await provider.generate(messages, profile=LLMProfiles.CHAT)

            # Удаляем любые левые ссылки, которые могла придумать LLM
            content = re.sub(
                r'<a\s+[^>]*href=["\'][^"\']+["\'][^>]*>(.*?)</a>',
                r"\1",
                content,
                flags=re.IGNORECASE,
            )

            # Добавляем свои источники (максимум 3 штуки)
            lower_content = content.lower()
            not_found = (
                "не нашел информации" in lower_content or "не найдена" in lower_content
            )
            if valid_sources and not not_found:
                links_html = []
                for s in valid_sources[:3]:
                    url = str(s.get("url", ""))
                    if not url:
                        continue
                    title = str(s.get("title") or DEFAULT_SOURCE_TITLE)
                    safe_url = escape(normalize_url_for_messaging(url), quote=True)
                    links_html.append(f'<a href="{safe_url}">{escape(title)}</a>')

                content += "\n\n<b>Источники:</b>\n" + "\n".join(links_html)

            logger.info(f"[{session_id}] Received response from LLM.")
            logger.info(
                f"[{session_id}] - Raw response (first 500 chars): {content[:500]}..."
            )
            logger.info(f"[{session_id}] - Response length: {len(content)} characters")

            _spawn_bg(
                _save_log_to_db(
                    user_id=user_id,
                    session_id=session_id,
                    message_type="llm_response",
                    content=content[:2000],
                    message_metadata={
                        "response_length": len(content),
                        "provider": provider.__class__.__name__,
                    },
                )
            )

            content = re.sub(r"foundland", "", content, flags=re.DOTALL)
            content = _sanitize_telegram_html(content)

            if not content:
                logger.warning(
                    f"[{session_id}] LLM returned empty content after sanitization."
                )
                content = "Ответ не найден"
            else:
                logger.info(
                    f"[{session_id}] Final LLM response after sanitization "
                    f"(first 300 chars): {content[:300]}..."
                )
        except Exception as e:
            logger.warning(f"[{session_id}] Provider generation error: {e}")
            content = "LLM временно недоступна."

        # 9. Сохраняем ответ в Redis + PG в фоне — пользователю отвечаем сразу.
        _spawn_bg(
            redis_client.add_message(
                session_id, {"role": "assistant", "content": content}
            )
        )
        if user_id:
            _spawn_bg(_save_message_to_pg(user_id, session_id, message, content))

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
