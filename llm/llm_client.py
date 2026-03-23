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
Ты — официальный дружелюбный помощник-бот для абитуриентов НГУ
(Новосибирский государственный университет).

Правила поведения:
1. Имя и легкий диалог: Если пользователь здоровается или говорит о себе (например,
   называет своё имя), обязательно используй историю переписки, чтобы поддержать
   беседу и обратиться по имени.
2. Вопросы об НГУ: Ищи фактическую информацию ИСКЛЮЧИТЕЛЬНО в блоке
   "Контекст из базы знаний об НГУ" ниже. Если ответа там нет, ответь по общим знаниям
   об НГУ, но предупреди, что точных данных нет.
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
            intent_response = await intent_provider.generate(intent_messages)
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
            try:
                rag_query = f"{message}\n\n{LIGHTRAG_FORMAT_HINT}"
                rag_context_raw, rag_sources = await query_graph_with_sources(rag_query)
                if not rag_context_raw or rag_context_raw.startswith(
                    "Error executing query"
                ):
                    rag_context = "Релевантный контекст из базы знаний не найден."
                    rag_sources = []
                else:
                    rag_context = rag_context_raw
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
                    '<a href="URL_1">URL_1</a>\n'
                    '<a href="URL_2">URL_2</a>\n\n'
                    "(Но если ты просто здороваешься, говоришь на отвлеченные темы или"
                    + " не нашел ответа в Контексте - блок "
                    + "'Источники:' ВООБЩЕ НЕ добавляй!)\n"
                    "КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ придумывать свои ссылки или брать их из истории переписки. "
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
                    entry_clean = re.sub(r"<a\s+href=[^>]+>.*?</a>", "", entry_clean).strip()
                    messages.append(AIMessage(content=entry_clean))

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
