import asyncio
import logging
from dataclasses import dataclass

from aiogram import Bot as TelegramBot
from maxapi import Bot as MaxBot

from bootstrap.runtime_settings import runtime_settings
from bot.notifications import send_pending_notifications
from config.instance import enabled, get_setting, revision
from config.university import is_nsu
from db.postgres.db import AsyncSessionLocal
from db.postgres.dto import RatingEntry
from db.postgres.services.rating import RatingService
from logging_config import setup_logging
from parser.rating import (
    ListFetchResult,
    RatingListFetcher,
    fetch_all_leaderboard_urls,
    parse_mock_rating_page,
)

logger = logging.getLogger(__name__)

# Границы адаптивного интервала опроса (сек).
POLL_MIN_SECONDS = int(get_setting("RATING_POLL_MIN_SECONDS", "300"))  # 5 минут
POLL_MAX_SECONDS = int(get_setting("RATING_POLL_MAX_SECONDS", "1800"))  # 30 минут
POLL_START_SECONDS = int(get_setting("RATING_POLL_START_SECONDS", "1800"))
# Пауза между запросами к сайту (вежливость к abiturient.nsu.ru).
REQUEST_DELAY_SECONDS = float(get_setting("RATING_REQUEST_DELAY", "0.3"))
RATING_DEGREE = get_setting("RATING_DEGREE", "bachelor")
# Доля списков с ошибкой, выше которой цикл считается неудачным: ERROR в логе
# и интервал опроса не растёт. Единичные сбои — WARNING.
FAILED_LISTS_THRESHOLD = float(get_setting("RATING_FAILED_LISTS_THRESHOLD", "0.2"))
# Сколько раз за обход можно перезапросить сессию и CSRF-токен после ошибки.
MAX_SESSION_REFRESHES = int(get_setting("RATING_MAX_SESSION_REFRESHES", "3"))


def _use_mock() -> bool:
    return get_setting("RATING_USE_MOCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class SyncResult:
    """Итог одного цикла синхронизации."""

    changed: bool = False
    # Цикл не справился: упал, не получил список URL или токен, либо доля
    # списков с ошибкой выше порога. Такой цикл не повод опрашивать реже.
    failed: bool = False
    total_lists: int = 0
    failed_lists: int = 0


def compute_next_interval(
    current: int,
    changed: bool,
    min_s: int = POLL_MIN_SECONDS,
    max_s: int = POLL_MAX_SECONDS,
    failed: bool = False,
) -> int:
    """Адаптивный интервал опроса.

    Были изменения → опрашиваем чаще (текущий ÷2); не было → реже (×2).
    Неудачный цикл ничего не говорит о том, менялись ли списки, поэтому
    интервал после него не растёт. Результат ограничен диапазоном [min_s, max_s].
    """
    if changed:
        nxt = current // 2
    elif failed:
        nxt = current
    else:
        nxt = current * 2
    return max(min_s, min(max_s, nxt))


async def _process_leaderboard(
    rating_service: RatingService,
    url: str,
    entries: list[RatingEntry],
    page_hash: str,
    direction_name: str,
) -> bool:
    """Обновляет один конкурсный список в БД.

    Места, новый хэш и уведомления в notification_outbox пишутся одним
    коммитом: либо всё, либо ничего. Возвращает, были ли изменения.
    """
    leaderboard = await rating_service.get_or_create_leaderboard(
        url, direction_name=direction_name
    )

    # Хэш совпал — данные не менялись, пропускаем.
    if leaderboard.content_hash == page_hash:
        return False

    # Дедуп по applicant_id: один человек может быть в нескольких номинациях —
    # оставляем лучшее (наименьшее) место.
    unique: dict[str, RatingEntry] = {}
    for entry in entries:
        ident = entry.identifier
        if not ident:
            continue
        if ident not in unique or entry.place < unique[ident].place:
            unique[ident] = entry

    stats = await rating_service.update_ratings_from_entries(
        leaderboard_id=leaderboard.id,
        entries=list(unique.values()),
        content_hash=page_hash or None,
        enqueue_notifications=True,
    )
    logger.info(
        "Список %s обновлён: уведомлений в очереди %d",
        url,
        len(stats.get("notifications", [])),
    )
    return True


async def _deliver_pending(bot: TelegramBot | None, max_client: MaxBot | None) -> None:
    """Рассылка из очереди; её сбой не должен ронять обход списков."""
    try:
        await send_pending_notifications(bot, max_client=max_client)
    except Exception as e:
        logger.error("Ошибка рассылки уведомлений из очереди: %s", e, exc_info=True)


def _report_failures(result: SyncResult) -> None:
    if not result.failed_lists:
        return
    share = result.failed_lists / result.total_lists
    if share > FAILED_LISTS_THRESHOLD:
        result.failed = True
        logger.error(
            "Не загрузились %d из %d конкурсных списков (%.0f%%, порог %.0f%%): "
            "данные этого цикла неполные, хэши упавших списков не обновлены.",
            result.failed_lists,
            result.total_lists,
            share * 100,
            FAILED_LISTS_THRESHOLD * 100,
        )
    else:
        logger.warning(
            "Не загрузились %d из %d конкурсных списков; хэши этих списков "
            "не обновлены, повторим в следующем цикле.",
            result.failed_lists,
            result.total_lists,
        )


async def run_sync_job(
    bot: TelegramBot | None, max_client: MaxBot | None = None
) -> SyncResult:
    """Один прогон: обход всех конкурсных списков → обновление БД → уведомления.

    Уведомления не копятся в памяти до конца обхода, а лежат в
    notification_outbox: рассылается то, что осталось с прошлых циклов, и
    новое — сразу после каждого изменившегося списка.
    """
    if not is_nsu():
        return SyncResult()
    logger.info("Начало цикла синхронизации рейтингов...")
    result = SyncResult()

    # Хвост прошлого цикла, если он упал между записью в базу и рассылкой.
    await _deliver_pending(bot, max_client)

    fetcher: RatingListFetcher | None = None
    try:
        if _use_mock():
            urls = [
                get_setting(
                    "MOCK_RATING_URL", "http://host.docker.internal:3000/api/rating"
                )
            ]
        else:
            urls = fetch_all_leaderboard_urls(RATING_DEGREE)

        if not urls:
            logger.error("Список URL конкурсных списков пуст — пропускаем итерацию.")
            result.failed = True
            return result

        logger.info("К обходу %d конкурсных списков", len(urls))
        result.total_lists = len(urls)

        if not _use_mock():
            fetcher = RatingListFetcher(max_refreshes=MAX_SESSION_REFRESHES)
            if not fetcher.open():
                logger.error("Не удалось получить CSRF-токен — пропускаем итерацию.")
                result.failed = True
                return result

        async with AsyncSessionLocal() as db_session:
            rating_service = RatingService(db_session)
            for url in urls:
                fetched: ListFetchResult
                if fetcher is None:
                    fetched = parse_mock_rating_page(url)
                else:
                    fetched = fetcher.fetch(url)
                    if REQUEST_DELAY_SECONDS > 0:
                        await asyncio.sleep(REQUEST_DELAY_SECONDS)

                if not fetched.ok:
                    # Хэш не трогаем: следующий цикл должен увидеть список заново.
                    result.failed_lists += 1
                    continue
                if not fetched.entries:
                    continue

                changed = await _process_leaderboard(
                    rating_service,
                    url,
                    fetched.entries,
                    fetched.page_hash,
                    fetched.direction_name,
                )
                if changed:
                    result.changed = True
                    await _deliver_pending(bot, max_client)

        _report_failures(result)
        logger.info(
            "Цикл синхронизации завершён. Изменения: %s, ошибок списков: %d из %d",
            result.changed,
            result.failed_lists,
            result.total_lists,
        )
    except Exception as e:
        result.failed = True
        logger.error(f"Ошибка во время цикла синхронизации: {e}", exc_info=True)
    finally:
        if fetcher is not None:
            fetcher.close()

    return result


def reload_settings() -> None:
    global POLL_MIN_SECONDS, POLL_MAX_SECONDS, REQUEST_DELAY_SECONDS
    global RATING_DEGREE, FAILED_LISTS_THRESHOLD, MAX_SESSION_REFRESHES
    POLL_MIN_SECONDS = int(get_setting("RATING_POLL_MIN_SECONDS", "300"))
    POLL_MAX_SECONDS = int(get_setting("RATING_POLL_MAX_SECONDS", "1800"))
    REQUEST_DELAY_SECONDS = float(get_setting("RATING_REQUEST_DELAY", "0.3"))
    RATING_DEGREE = get_setting("RATING_DEGREE", "bachelor")
    FAILED_LISTS_THRESHOLD = float(get_setting("RATING_FAILED_LISTS_THRESHOLD", "0.2"))
    MAX_SESSION_REFRESHES = int(get_setting("RATING_MAX_SESSION_REFRESHES", "3"))


async def main() -> None:
    async with runtime_settings():
        interval = int(get_setting("RATING_POLL_START_SECONDS", "1800"))
        while True:
            reload_settings()
            bot = (
                TelegramBot(token=get_setting("BOT_TOKEN", ""))
                if enabled("TELEGRAM_ENABLED")
                else None
            )
            max_client = (
                MaxBot(token=get_setting("MAX_BOT_TOKEN", ""))
                if enabled("MAX_ENABLED")
                else None
            )
            observed_revision = revision()
            try:
                result = (
                    await run_sync_job(bot, max_client=max_client)
                    if is_nsu() and enabled("SCHEDULER_ENABLED") and (bot or max_client)
                    else SyncResult()
                )
                interval = compute_next_interval(
                    interval,
                    result.changed,
                    min_s=POLL_MIN_SECONDS,
                    max_s=POLL_MAX_SECONDS,
                    failed=result.failed,
                )
            finally:
                if bot:
                    await bot.session.close()
                if max_client:
                    await max_client.close_session()
            # A configuration edit wakes the scheduler without a process restart.
            for _ in range(max(1, interval // 2)):
                await asyncio.sleep(2)
                if revision() != observed_revision:
                    break


if __name__ == "__main__":
    # Здесь, а не на импорте: тесты импортируют модуль и не должны писать в logs/.
    setup_logging()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Работа планировщика остановлена вручную.")
