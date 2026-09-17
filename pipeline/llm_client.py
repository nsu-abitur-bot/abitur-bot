import asyncio
import json
import logging
import re
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from html import unescape
from typing import Awaitable, Callable, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from abbrev.expander import get_abbrev_expander
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.message import MessageService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.topic import TopicService
from db.postgres.services.user import UserService
from db.redis.client import RedisClient
from faq.matcher import get_faq_matcher
from llm.base import LLMUsage
from llm.factory import get_llm_provider
from llm.profiles import LLMProfiles
from pipeline.tools import ADMISSION_SCORES_TOOL, default_tool_executor
from rag.crag import load_crag_config
from rag.retriever import query_graph_with_crag, query_graph_with_sources

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LlmAnswer:
    """Ответ пайплайна: текст и источники, без разметки конкретного канала.

    Разметку накладывает адаптер канала (`bot/formatting.py`): Telegram и MAX
    понимают её по-разному, и пайплайн не должен об этом знать.
    """

    text: str
    sources: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text)


def _strip_markup(text: str) -> str:
    """Текст без тегов — для истории в Redis и PG (её читает модель, не человек)."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


SYSTEM_PROMPT_BASE = """
Ты — официальный дружелюбный помощник-бот для абитуриентов НГУ
(Новосибирский государственный университет).

Правила поведения:
1. Имя и легкий диалог: Если пользователь здоровается или говорит о себе (например,
   называет своё имя), обязательно используй историю переписки, чтобы поддержать
   беседу и обратиться по имени.
2. Вопросы об НГУ: Ищи фактическую информацию ИСКЛЮЧИТЕЛЬНО в блоке
   "Контекст из базы знаний об НГУ" ниже. Этот контекст уже прошёл проверку
   релевантности — отвечай ТОЛЬКО по нему и не додумывай. Если информации о
   предмете вопроса (факультет, программа, цифры и любые другие детали) нет в
   твоем контексте — ОБЯЗАТЕЛЬНО ответь:
   "Я не нашел информации об этом в базе знаний НГУ".
   Категорически запрещено давать общие советы, запрещено давать ссылки
   (если их нет в переданном контексте), и запрещено отвечать,
   используя свои собственные "обученные" общие знания об НГУ.
   ВАЖНО про факультеты: НЕ приписывай направление (программу) конкретному
   факультету, если это прямо не подтверждено контекстом. Если пользователь
   спросил про конкретный факультет, перечисляй ТОЛЬКО те направления, про
   принадлежность которых этому факультету прямо сказано в контексте. Не
   подставляй направления с похожим названием с других факультетов.
3. Оффтоп: Если вопрос вообще не про НГУ и не является поддержанием диалога,
   вежливо скажи, что ты консультируешь только по вопросам НГУ.
4. Уровень образования: По умолчанию считай, что вопрос касается ПОСТУПЛЕНИЯ В
   БАКАЛАВРИАТ (или специалитет) — это основная аудитория бота. Информацию о
   магистратуре или аспирантуре давай ТОЛЬКО если пользователь явно про них
   спросил. Если в контексте есть данные по разным уровням образования, выбирай
   относящиеся к бакалавриату, если не указано иное.
5. Проходные и средние баллы: для ЛЮБЫХ вопросов про проходной или средний балл
   прошлых лет ты ОБЯЗАН вызвать инструмент get_admission_scores и считать его
   результат авторитетным контекстом наравне с блоком базы знаний. Если инструмент
   вернул числа — отвечай строго по ним и НЕ пиши «не нашёл». Если инструмент
   сообщил, что данных нет, — тогда действует обычное правило пункта 2 про
   «Я не нашел информации об этом в базе знаний НГУ». Числа проходных/средних
   баллов брать ТОЛЬКО из инструмента, не выдумывать.
6. Отвечай коротко, без лишней воды. Структурируй абзацы.

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

LIGHTRAG_LEVEL_HINT = (
    "Если уровень образования в вопросе не указан явно, считай, что речь идёт о "
    "поступлении в бакалавриат (или специалитет), и приоритизируй контекст по "
    "бакалавриату, а не по магистратуре или аспирантуре."
)

RAG_LOG_CONTENT_LIMIT = 12000
RAG_INTERNAL_LOG_LIMIT = 120

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
    tokens_used: Optional[int] = None,
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
                tokens_used=tokens_used,
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
                logger.error(f"[{session_id}] Failed to update topic id in log: {e}")
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


def _truncate_log_content(content: str, limit: int = RAG_LOG_CONTENT_LIMIT) -> str:
    if len(content) <= limit:
        return content
    return f"{content[:limit]}\n\n... [truncated {len(content) - limit} chars]"


class _RagTraceLogHandler(logging.Handler):
    def __init__(self, limit: int = RAG_INTERNAL_LOG_LIMIT) -> None:
        super().__init__(level=logging.INFO)
        self.limit = limit
        self.lines: list[str] = []
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = f"{record.levelname}: {record.getMessage()}"
        except Exception:
            return
        if len(self.lines) < self.limit:
            self.lines.append(line)
        else:
            self.dropped += 1

    def get_lines(self) -> list[str]:
        if self.dropped <= 0:
            return self.lines
        return [*self.lines, f"... [truncated {self.dropped} internal RAG log lines]"]


@contextmanager
def _capture_lightrag_logs():
    lightrag_logger = logging.getLogger("lightrag")
    handler = _RagTraceLogHandler()
    previous_level = lightrag_logger.level
    previous_propagate = lightrag_logger.propagate
    lightrag_logger.setLevel(logging.INFO)
    lightrag_logger.propagate = True
    lightrag_logger.addHandler(handler)
    try:
        yield handler
    finally:
        lightrag_logger.removeHandler(handler)
        lightrag_logger.setLevel(previous_level)
        lightrag_logger.propagate = previous_propagate


def _format_rag_response_log(context: str, trace_lines: list[str]) -> str:
    blocks = []
    if trace_lines:
        blocks.append("Трассировка поиска в базе знаний:\n" + "\n".join(trace_lines))
    blocks.append("Ответ базы знаний:\n" + context)
    return _truncate_log_content("\n\n".join(blocks))


# Блок «Источники» в прошлых ответах ассистента: в историю он попадал целиком,
# и модель начинала копировать ссылки из него.
_SOURCES_BLOCK_RE = re.compile(
    r"\n?(?:<br>|<b>|###\s*|\*+\s*)*\s*"
    r"(?:Источники|Источник|References|Ссылки)(?:\s+информации)?:?\s*"
    r"(?:</b>|\*+)*\s*(?:\n|<a)[\s\S]*",
    re.IGNORECASE,
)
# Ссылка целиком — вырезаем из истории.
_ANCHOR_RE = re.compile(r"<a\s+href=[^>]+>.*?</a>")
# Ссылка, от которой оставляем только текст: свои источники приклеивает канал.
_ANCHOR_TEXT_RE = re.compile(
    r"<a\s+[^>]*href=[\"\'][^\"\']+[\"\'][^>]*>(.*?)</a>",
    re.IGNORECASE,
)


StreamCallback = Callable[[str], Awaitable[None]]
StatusCallback = Callable[[str], Awaitable[None]]

STATUS_FAQ_LOOKUP = "🔎 Поиск готового ответа…"
STATUS_INTENT = "🧭 Анализ вопроса…"
STATUS_RAG = "📚 Поиск в базе знаний…"
STATUS_GENERATING = "✍️ Готовлю ответ…"


@dataclass
class _Ctx:
    """То, что нужно каждому шагу: кому отвечаем и куда писать логи."""

    session_id: str
    user_id: int
    log_entry_id: Optional[int] = None
    status_callback: Optional[StatusCallback] = None

    async def emit_status(self, text: str) -> None:
        """Промежуточный статус пользователю. Сбой колбэка не ломает ответ."""
        if self.status_callback is None:
            return
        try:
            await self.status_callback(text)
        except Exception as exc:
            logger.warning(f"[{self.session_id}] status_callback error: {exc}")

    def log(self, message_type: str, content: str, **metadata) -> None:
        """Запись в аналитический лог, fire-and-forget."""
        tokens_used = metadata.pop("tokens_used", None)
        _spawn_bg(
            _save_log_to_db(
                user_id=self.user_id,
                session_id=self.session_id,
                message_type=message_type,
                content=content,
                message_metadata=metadata,
                tokens_used=tokens_used,
            )
        )


@dataclass
class _RagResult:
    """Что база знаний дала для промпта."""

    context: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class _Postprocessed:
    """Готовый ответ и то, что от него пишем в лог.

    `log_text` отличается от `text`: в лог идёт ответ до вычистки артефактов,
    чтобы по логу было видно, что именно вернула модель.
    """

    text: str
    sources: list[dict]
    log_text: str


def _expand_abbrevs(message: str, session_id: str) -> str:
    """Раскрывает аббревиатуры: «ФИТ» → «ФИТ (Факультет информационных...)».

    Нужно и FAQ-матчеру, и поиску: без раскрытия «ФИТ» не находит документы,
    где написано полное название.
    """
    try:
        return get_abbrev_expander().expand(message)
    except Exception as exc:
        logger.warning(f"[{session_id}] abbrev expander error: {exc}")
        return message


async def _await_faq(
    task: "asyncio.Task[Optional[str]]", session_id: str
) -> Optional[str]:
    try:
        return await task
    except Exception as e:
        logger.warning(f"[{session_id}] FAQ matcher error: {e}")
        return None


async def _await_history(task: "asyncio.Task[list]", session_id: str) -> list:
    try:
        return await task
    except Exception as e:
        logger.warning(f"[{session_id}] Redis get_history error: {e}")
        return []


async def _retrieve_context(
    expanded_message: str,
    history_text: str,
    ctx: _Ctx,
) -> _RagResult:
    """Ищет контекст в базе знаний. При любом сбое возвращает текст-заглушку.

    Заглушка попадает в системный промпт, и модель по ней понимает, что
    опираться не на что. Поднимать исключение нельзя: без базы знаний бот всё
    равно должен ответить хотя бы «не знаю».
    """
    session_id = ctx.session_id
    rag_trace_lines: list[str] = []
    try:
        logger.info(f"[{session_id}] Querying LightRAG for context.")
        rag_query = (
            f"{expanded_message}\n\n{LIGHTRAG_LEVEL_HINT}\n\n{LIGHTRAG_FORMAT_HINT}"
        )
        ctx.log(
            "rag_query",
            _truncate_log_content(rag_query),
            title="Запрос к базе знаний",
            query_length=len(rag_query),
            history_present=bool(history_text),
        )

        use_crag = (await load_crag_config()).enabled
        with _capture_lightrag_logs() as rag_trace:
            if use_crag:
                logger.info(f"[{session_id}] CRAG enabled — using corrective RAG.")
                raw_context, metadata_sources = await query_graph_with_crag(
                    rag_query,
                    conversation_history=history_text or None,
                )
            else:
                raw_context, metadata_sources = await query_graph_with_sources(
                    rag_query,
                    conversation_history=history_text or None,
                )
        rag_trace_lines = rag_trace.get_lines()

        found = bool(raw_context) and not raw_context.startswith("Error executing query")
        if found:
            sources = metadata_sources
            context = raw_context
            logger.info(
                f"[{session_id}] Retrieved context from RAG (sources: {len(sources)})."
            )
            logger.info(f"[{session_id}] - Context (first 500 chars): {context[:500]}...")
            logger.info(f"[{session_id}] - Sources ({len(sources)}): {sources}")
        else:
            logger.info(f"[{session_id}] No relevant context found in RAG.")
            sources = []
            context = "Релевантный контекст из базы знаний не найден."

        ctx.log(
            "rag_response",
            _format_rag_response_log(raw_context or context, rag_trace_lines),
            title="Ответ базы знаний",
            sources=sources,
            context_length=len(raw_context or context),
            sources_count=len(sources),
            internal_logs_count=len(rag_trace_lines),
            found_context=found,
        )
        return _RagResult(context=_clean_rag_context(context), sources=sources)
    except Exception as e:
        logger.warning(f"[{session_id}] LightRAG query error: {e}")
        context = "База знаний временно недоступна."
        ctx.log(
            "rag_response",
            context,
            title="Ответ базы знаний",
            error=str(e),
            internal_logs_count=len(rag_trace_lines),
            found_context=False,
        )
        return _RagResult(context=context)


def _build_messages(rag_context: str, history_entries: list[dict]) -> list[BaseMessage]:
    """Собирает промпт: системная часть с контекстом плюс история переписки.

    Из прошлых ответов ассистента вырезаем блок «Источники» и ссылки: раньше
    они попадали в историю целиком и модель начинала их копировать.
    """
    sources_hint = (
        "\n\nИНСТРУКЦИЯ К ОТВЕТУ:\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать блок 'Источники' или перечислять ссылки. "
        "Просто ответь на вопрос пользователя, опираясь на контекст!"
    )
    system_prompt = SYSTEM_PROMPT_BASE.format(
        context=rag_context, sources_hint=sources_hint
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    for entry in history_entries:
        role = entry.get("role", "")
        entry_content = entry.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=entry_content))
        elif role == "assistant":
            entry_clean = _SOURCES_BLOCK_RE.sub("", entry_content)
            entry_clean = _ANCHOR_RE.sub("", entry_clean).strip()
            messages.append(AIMessage(content=entry_clean))

    return messages


async def _run_tool_loop(
    messages: list[BaseMessage],
    session_id: str,
    stream_callback: Optional[StreamCallback] = None,
) -> tuple[str, LLMUsage, str]:
    """Генерация ответа с доступом к инструментам. Возвращает текст, расход, провайдера.

    Со стримингом отдаём накопленный текст в транспорт по мере генерации; сбой
    колбэка не должен ломать сам ответ.
    """
    provider = get_llm_provider()
    provider_name = provider.__class__.__name__
    logger.info(f"[{session_id}] Sending payload to LLM ({provider_name}).")

    if stream_callback is None:
        result = await provider.generate_with_tools(
            messages,
            tools=[ADMISSION_SCORES_TOOL],
            tool_executor=default_tool_executor,
            profile=LLMProfiles.CHAT,
        )
        return result.text, result.usage, provider_name

    streamed = ""

    async def on_delta(delta: str) -> None:
        nonlocal streamed
        if not delta:
            return
        streamed += delta
        try:
            await stream_callback(streamed)
        except Exception as cb_exc:
            logger.warning(f"[{session_id}] stream_callback error: {cb_exc}")

    result = await provider.generate_with_tools(
        messages,
        tools=[ADMISSION_SCORES_TOOL],
        tool_executor=default_tool_executor,
        profile=LLMProfiles.CHAT,
        on_delta=on_delta,
    )
    return result.text or streamed.strip(), result.usage, provider_name


def _postprocess(content: str, rag_sources: list[dict]) -> _Postprocessed:
    """Готовит ответ к отправке: ссылки, источники, артефакты, пустой ответ."""
    # Ссылки, которые придумала модель, оставляем текстом без адреса: свои
    # источники приклеивает канал, из метаданных базы знаний.
    content = _ANCHOR_TEXT_RE.sub(r"\1", content)
    log_text = content

    # К отказу «не нашёл информации» список источников приклеивать бессмысленно.
    lowered = content.lower()
    not_found = (
        "не нашел информации" in lowered
        or "не нашёл информации" in lowered
        or "не найдена" in lowered
    )
    sources = [] if not_found else list(rag_sources)

    content = content.strip()
    if not content:
        logger.warning("LLM returned empty content.")
        return _Postprocessed(text="Ответ не найден", sources=[], log_text=log_text)

    return _Postprocessed(text=content, sources=sources, log_text=log_text)


async def _answer_from_faq(
    faq_answer: str,
    message: str,
    redis_client: RedisClient,
    ctx: _Ctx,
) -> LlmAnswer:
    """Готовый ответ из матчера частых вопросов: модель не зовём вообще."""
    session_id = ctx.session_id
    logger.info(f"[{session_id}] FAQ match found, returning predefined answer.")

    ctx.log("faq_match", faq_answer, source="faq")
    _spawn_bg(
        redis_client.add_message(
            session_id, {"role": "assistant", "content": faq_answer}
        )
    )
    if ctx.user_id:
        _spawn_bg(_save_message_to_pg(ctx.user_id, session_id, message, faq_answer))
    return LlmAnswer(text=faq_answer)


async def _generate_answer(
    rag: _RagResult,
    history_entries: list[dict],
    ctx: _Ctx,
    stream_callback: Optional[StreamCallback] = None,
) -> tuple[str, list[dict]]:
    """Промпт, генерация, постобработка. При сбое модели — текст-заглушка."""
    session_id = ctx.session_id
    try:
        logger.info(f"[{session_id}] Preparing prompt to LLM provider.")
        messages = _build_messages(rag.context, history_entries)

        await ctx.emit_status(STATUS_GENERATING)
        content, llm_usage, provider_name = await _run_tool_loop(
            messages, session_id, stream_callback
        )

        processed = _postprocess(content, rag.sources)
        logger.info(f"[{session_id}] Received response from LLM.")
        logger.info(
            f"[{session_id}] - Raw response (first 500 chars): "
            f"{processed.log_text[:500]}..."
        )
        ctx.log(
            "llm_response",
            processed.log_text[:2000],
            response_length=len(processed.log_text),
            provider=provider_name,
            tokens=llm_usage.to_dict(),
            tokens_used=llm_usage.total_tokens or None,
        )
        logger.info(
            f"[{session_id}] Final LLM response "
            f"(first 300 chars): {processed.text[:300]}..."
        )
        return processed.text, processed.sources
    except Exception as e:
        logger.warning(f"[{session_id}] Provider generation error: {e}")
        return "LLM временно недоступна.", []


def _store_answer(
    content: str,
    message: str,
    redis_client: RedisClient,
    ctx: _Ctx,
) -> None:
    """Сохраняет ответ в историю в фоне — пользователю отвечаем сразу.

    В историю кладём текст без тегов: её читает модель, а не человек.
    """
    stored_text = _strip_markup(content)
    _spawn_bg(
        redis_client.add_message(
            ctx.session_id, {"role": "assistant", "content": stored_text}
        )
    )
    if ctx.user_id:
        _spawn_bg(_save_message_to_pg(ctx.user_id, ctx.session_id, message, stored_text))


async def ask_local_llm(
    message: str,
    session_id: str,
    user_id: int = 0,
    log_entry_id: Optional[int] = None,
    stream_callback: Optional[StreamCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> LlmAnswer:
    """Отвечает на сообщение пользователя. Порядок шагов — в теле функции.

    session_id — идентификатор переписки, по нему берётся и сохраняется история.
    user_id — внутренний идентификатор для записей в PG.
    stream_callback вызывается с накопленным текстом по мере генерации: на
    FAQ-матче модель не зовётся, и колбэк не срабатывает.
    """
    logger.info(
        f"[{session_id}] New message received from user={user_id}: {message[:50]}..."
    )
    ctx = _Ctx(
        session_id=session_id,
        user_id=user_id,
        log_entry_id=log_entry_id,
        status_callback=status_callback,
    )

    try:
        redis_client = await get_redis_client()
        await redis_client.add_message(session_id, {"role": "user", "content": message})

        expanded_message = _expand_abbrevs(message, session_id)

        # Классификация темы нужна только аналитическому логу и на ответ не
        # влияет — уезжает в фон.
        _spawn_bg(_classify_intent_bg(expanded_message, log_entry_id, session_id))

        await ctx.emit_status(STATUS_FAQ_LOOKUP)

        # FAQ и история стартуют параллельно: FAQ отвечает за 0.3-1с, история
        # за ~10мс. Поиск в базе знаний нельзя запускать до результата FAQ.
        faq_matcher = get_faq_matcher()
        task_faq = asyncio.create_task(faq_matcher.match_async(expanded_message))
        task_history = asyncio.create_task(redis_client.get_history(session_id))

        faq_answer = await _await_faq(task_faq, session_id)
        if faq_answer:
            task_history.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task_history
            return await _answer_from_faq(faq_answer, message, redis_client, ctx)

        history_entries = await _await_history(task_history, session_id)
        history_text = _build_history_text(history_entries)

        await ctx.emit_status(STATUS_RAG)
        rag = await _retrieve_context(expanded_message, history_text, ctx)

        content, answer_sources = await _generate_answer(
            rag, history_entries, ctx, stream_callback
        )
        _store_answer(content, message, redis_client, ctx)

        logger.info(f"[{session_id}] Message processing complete.")
        return LlmAnswer(text=content, sources=answer_sources)

    except Exception as e:
        logger.error(f"[{session_id}] LLM error: {e}")
        return LlmAnswer(text="Что-то пошло не так")


async def cleanup_redis():
    """Закрывает Redis соединение при завершении работы."""
    global _redis_client
    if _redis_client is not None:
        with suppress(Exception):
            await _redis_client.close()
        _redis_client = None
