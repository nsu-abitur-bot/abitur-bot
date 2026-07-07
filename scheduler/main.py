import asyncio
import logging
import os

from aiogram import Bot as TelegramBot
from maxapi import Bot as MaxBot

from bot.notifications import notify_users
from db.postgres.db import AsyncSessionLocal
from db.postgres.dto import RatingEntry
from db.postgres.services.rating import RatingService
from parser.rating import (
    create_rating_session,
    fetch_all_leaderboard_urls,
    parse_mock_rating_page,
    parse_rating_url_with_session,
)

logger = logging.getLogger(__name__)

# Границы адаптивного интервала опроса (сек).
POLL_MIN_SECONDS = int(os.getenv("RATING_POLL_MIN_SECONDS", "300"))  # 5 минут
POLL_MAX_SECONDS = int(os.getenv("RATING_POLL_MAX_SECONDS", "1800"))  # 30 минут
POLL_START_SECONDS = int(os.getenv("RATING_POLL_START_SECONDS", "1800"))
# Пауза между запросами к сайту (вежливость к abiturient.nsu.ru).
REQUEST_DELAY_SECONDS = float(os.getenv("RATING_REQUEST_DELAY", "0.3"))
RATING_DEGREE = os.getenv("RATING_DEGREE", "bachelor")


def _use_mock() -> bool:
    return os.getenv("RATING_USE_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def compute_next_interval(
    current: int,
    changed: bool,
    min_s: int = POLL_MIN_SECONDS,
    max_s: int = POLL_MAX_SECONDS,
) -> int:
    """Адаптивный интервал опроса.

    Были изменения → опрашиваем чаще (текущий ÷2); не было → реже (×2).
    Результат ограничен диапазоном [min_s, max_s].
    """
    nxt = current // 2 if changed else current * 2
    return max(min_s, min(max_s, nxt))


async def _process_leaderboard(
    rating_service: RatingService,
    url: str,
    entries: list[RatingEntry],
    page_hash: str,
    direction_name: str,
) -> tuple[bool, list]:
    """Обновляет один конкурсный список в БД.

    Возвращает (были ли изменения, список уведомлений).
    """
    leaderboard = await rating_service.get_or_create_leaderboard(
        url, direction_name=direction_name
    )

    # Хэш совпал — данные не менялись, пропускаем.
    if leaderboard.content_hash == page_hash:
        return False, []

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
        leaderboard_id=leaderboard.id, entries=list(unique.values())
    )
    if page_hash:
        await rating_service.update_leaderboard_hash(leaderboard.id, page_hash)

    return True, stats.get("notifications", [])


async def run_sync_job(bot: TelegramBot | None, max_client: MaxBot | None = None) -> bool:
    """Один прогон: обход всех конкурсных списков → обновление БД → уведомления.

    Возвращает True, если хотя бы в одном списке были изменения (для адаптивного
    интервала).
    """
    logger.info("Начало цикла синхронизации рейтингов...")
    changed_any = False
    all_notifications: list = []

    try:
        if _use_mock():
            urls = [
                os.getenv(
                    "MOCK_RATING_URL", "http://host.docker.internal:3000/api/rating"
                )
            ]
        else:
            urls = fetch_all_leaderboard_urls(RATING_DEGREE)

        if not urls:
            logger.warning("Список URL конкурсных списков пуст — пропускаем итерацию.")
            return False

        logger.info("К обходу %d конкурсных списков", len(urls))

        session = None
        csrf_token = None
        if not _use_mock():
            session, csrf_token = create_rating_session()
            if not session or not csrf_token:
                logger.warning("Не удалось получить CSRF-токен — пропускаем итерацию.")
                return False

        try:
            async with AsyncSessionLocal() as db_session:
                rating_service = RatingService(db_session)
                for url in urls:
                    if _use_mock():
                        entries, page_hash, direction_name = parse_mock_rating_page(url)
                    else:
                        assert session is not None and csrf_token is not None
                        entries, page_hash, direction_name = (
                            parse_rating_url_with_session(session, csrf_token, url)
                        )
                        if REQUEST_DELAY_SECONDS > 0:
                            await asyncio.sleep(REQUEST_DELAY_SECONDS)

                    if not entries:
                        continue

                    changed, notifications = await _process_leaderboard(
                        rating_service, url, entries, page_hash, direction_name
                    )
                    if changed:
                        changed_any = True
                        all_notifications.extend(notifications)
        finally:
            if session is not None:
                session.close()

        if all_notifications:
            logger.info("Рассылка %d уведомлений...", len(all_notifications))
            await notify_users(bot, all_notifications, max_client=max_client)
        else:
            logger.info("Изменений нет, уведомления не требуются.")

        logger.info("Цикл синхронизации завершён. Изменения: %s", changed_any)
    except Exception as e:
        logger.error(f"Ошибка во время цикла синхронизации: {e}", exc_info=True)

    return changed_any


async def main():
    """Планировщик с адаптивным интервалом опроса."""
    bot_token = os.getenv("BOT_TOKEN")
    max_bot_token = os.getenv("MAX_BOT_TOKEN")
    if not bot_token:
        logger.warning(
            "Переменная окружения BOT_TOKEN не задана. Уведомления могут не работать!"
        )

    max_client = None
    if max_bot_token:
        try:
            max_client = MaxBot(token=max_bot_token)
            logger.info("MAX клиент для уведомлений инициализирован")
        except Exception as exc:
            logger.warning("Не удалось инициализировать MAX клиент: %s", exc)

    bot = TelegramBot(token=bot_token) if bot_token else None

    interval = max(POLL_MIN_SECONDS, min(POLL_MAX_SECONDS, POLL_START_SECONDS))
    logger.info("Запуск планировщика. Стартовый интервал: %d сек.", interval)

    while True:
        changed = False
        if bot is not None or max_client is not None:
            changed = await run_sync_job(bot, max_client=max_client)
        interval = compute_next_interval(interval, changed)
        logger.info(
            "Ожидание %d сек до следующего опроса (были изменения: %s)...",
            interval,
            changed,
        )
        await asyncio.sleep(interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Работа планировщика остановлена вручную.")
