"""Алерты дежурному: сервис упал или растёт доля ошибок (#243).

Проверки сознательно дешёвые и без своих зависимостей: проба живости API
(#241) и агрегат ошибок из метрик ответов (#242). Уведомление — Telegram
ботом в чат команды; повторы одного и того же проблемы подавляются
кулдауном, чтобы ночной инцидент не разбудил сотней сообщений.

Что делать по каждому алерту — docs/runbook.md: регламент рассчитан на
любого члена команды без знания внутренних деталей сервера.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from urllib.parse import quote

from aiogram import Bot

from config.instance import enabled, get_setting

logger = logging.getLogger(__name__)

RUNBOOK_URL = "https://github.com/nsu-abitur-bot/abitur-bot/blob/main/docs/runbook.md"

# (status, body) ответа HTTP; отдельный тип вместо Response, чтобы тесты
# подставляли фейк без сети.
Fetcher = Callable[[str], Awaitable[tuple[int, str]]]


@dataclass
class AlertState:
    """Кулдаун по ключу проблемы: первое срабатывание сразу, повторы — реже."""

    cooldown_seconds: float = 1800.0
    last_sent: dict[str, float] = field(default_factory=dict)

    def should_send(self, key: str, now: float) -> bool:
        sent_at = self.last_sent.get(key)
        if sent_at is not None and now - sent_at < self.cooldown_seconds:
            return False
        self.last_sent[key] = now
        return True


async def check_api_health(fetch: Fetcher, api_base: str) -> list[str]:
    try:
        status, _ = await fetch(f"{api_base}/health")
    except Exception as exc:
        return [f"API не отвечает ({type(exc).__name__}): проба живости упала"]
    if status != 200:
        return [f"API отвечает кодом {status} на /health"]
    return []


async def check_error_growth(fetch: Fetcher, api_base: str) -> list[str]:
    """Доля ошибочных ответов за окно выше порога.

    Минимум вопросов отсекает пустую базу и ночной штиль: 1 ошибка из
    2 вопросов — не рост ошибок, а шум.
    """
    threshold = float(get_setting("ALERT_ERROR_THRESHOLD", "0.2"))
    min_questions = int(get_setting("ALERT_MIN_QUESTIONS", "5"))
    window_minutes = int(get_setting("ALERT_WINDOW_MINUTES", "60"))

    # Окно с точностью до часовых бакетов ответа: для порога тревоги этого
    # достаточно, отдельный почасовой срез не нужен.
    start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    url = (
        f"{api_base}/api/v1/logs/answer-stats"
        f"?start={quote(start.isoformat())}&group_by=hour"
    )
    try:
        status, body = await fetch(url)
        if status != 200:
            return [f"Метрики ответов недоступны: /answer-stats вернул {status}"]
    except Exception as exc:
        return [f"Метрики ответов недоступны ({type(exc).__name__})"]

    try:
        stats = json.loads(body)
    except ValueError:
        return ["Метрики ответов недоступны: /answer-stats вернул не JSON"]

    questions = int(stats.get("total_questions") or 0)
    errors = int(stats.get("total_errors") or 0)
    if questions < min_questions:
        return []
    share = errors / questions
    if share > threshold:
        return [
            f"Доля ошибок за {window_minutes} мин: {errors} из {questions} "
            f"вопросов ({share:.0%}), порог {threshold:.0%}"
        ]
    return []


async def check_once(fetch: Fetcher, api_base: str) -> list[str]:
    problems: list[str] = []
    # Проверки независимы: сбой одной не должен прятать другую.
    problems.extend(await check_api_health(fetch, api_base))
    problems.extend(await check_error_growth(fetch, api_base))
    return problems


async def send_alert(text: str) -> None:
    """Одно уведомление в чат команды; свой Bot на отправку — сессия короткая."""
    chat_id = get_setting("ALERT_CHAT_ID", "")
    token = get_setting("ALERT_BOT_TOKEN", "") or get_setting("BOT_TOKEN", "")
    if not chat_id or not token:
        logger.warning("Алерту некуда идти: ALERT_CHAT_ID/токен не заданы")
        return
    bot = Bot(token=token)
    try:
        await bot.send_message(chat_id, text, disable_web_page_preview=True)
    finally:
        await bot.session.close()


async def _aiohttp_fetch(url: str) -> tuple[int, str]:
    import aiohttp

    async with (
        aiohttp.ClientSession() as session,
        session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response,
    ):
        return response.status, await response.text()


async def monitor_loop() -> None:
    """Цикл проверок; сбои самих проверок не роняют планировщик."""
    api_base = get_setting("ALERT_API_BASE", "http://127.0.0.1:8000")
    interval = int(get_setting("ALERT_INTERVAL_SECONDS", "60"))
    cooldown = float(get_setting("ALERT_REPEAT_MINUTES", "30")) * 60
    state = AlertState(cooldown_seconds=cooldown)

    while True:
        try:
            problems = await check_once(_aiohttp_fetch, api_base)
            now = time.monotonic()
            for problem in problems:
                if state.should_send(problem, now):
                    await send_alert(f"⚠️ {problem}\nРегламент: {RUNBOOK_URL}")
        except Exception:
            logger.exception("Цикл алертов упал; продолжаем")
        await asyncio.sleep(interval)


def monitor_enabled() -> bool:
    """Алерты включены явным флагом и имеют адресат."""
    return enabled("ALERT_ENABLED") and bool(get_setting("ALERT_CHAT_ID", ""))
