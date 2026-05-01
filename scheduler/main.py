import asyncio
import logging
import os

from aiogram import Bot as TelegramBot
from maxapi import Bot as MaxBot

from bot.notifications import notify_users
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.rating import RatingService
from parser.rating import (
    parse_mock_rating_page,
)

logger = logging.getLogger(__name__)


async def run_sync_job(bot: TelegramBot | None, max_client: MaxBot | None = None):
    """Скрипт одного прогона: парсинг -> обновление БД -> уведомления."""
    logger.info("Начало цикла синхронизации рейтингов...")
    try:
        url = os.getenv(
            "MOCK_RATING_URL", "http://host.docker.internal:3000/abitur-web/api/rating"
        )
        logger.info(f"Парсинг новых данных с мок-URL: {url}")

        parsed_entries, page_hash, direction_name = parse_mock_rating_page(url)

        if not parsed_entries:
            logger.warning("Не удалось получить данные с сайта. Пропускаем итерацию.")
            return

        # Шаг 2: Обновляем базу данных и получаем изменения
        logger.info("Обновление базы данных...")
        async with AsyncSessionLocal() as session:
            rating_service = RatingService(session)

            # Получим/создадим запись о рейтинге (Leaderboard) по URL
            leaderboard = await rating_service.get_or_create_leaderboard(
                url, direction_name=direction_name
            )
            leaderboard_id = leaderboard.id

            # Проверяем, изменились ли данные на странице
            if leaderboard.content_hash == page_hash:
                logger.info(
                    f"Данные не изменились (хэш совпадает: {page_hash})."
                    + " Пропускаем обновление БД."
                )
                return

            # Отфильтруем дубликаты по applicant_id, оставляя самое высокое место
            # (защита от мок-данных, в которых один человек может быть
            # в нескольких номинациях сразу)
            unique_entries = {}
            for entry in parsed_entries:
                ident = entry.identifier
                if not ident:
                    continue
                if ident not in unique_entries:
                    unique_entries[ident] = entry
                else:
                    # Оставляем лучшее место (наименьшее число)
                    if entry.place < unique_entries[ident].place:
                        unique_entries[ident] = entry

            deduplicated_entries = list(unique_entries.values())

            stats = await rating_service.update_ratings_from_entries(
                leaderboard_id=leaderboard_id, entries=deduplicated_entries
            )

            # Обновим хэш после успешного прогона
            if page_hash:
                await rating_service.update_leaderboard_hash(leaderboard_id, page_hash)

        # Получаем данные об уведомлениях
        notifications = stats.get("notifications", [])

        # Шаг 3: Рассылаем уведомления пользователям (если есть изменения)
        if notifications:
            logger.info(
                f"Рассылка уведомлений для {len(notifications)}"
                + "пользователей/изменений..."
            )
            await notify_users(bot, notifications, max_client=max_client)
        else:
            logger.info("Изменений нет, уведомления не требуются.")

        logger.info("Цикл синхронизации успешно завершен.")
    except Exception as e:
        logger.error(f"Ошибка во время цикла синхронизации: {e}", exc_info=True)


async def main():
    """Планировщик, который запускает задачу каждую N секунд."""
    interval = 10  # Интервал в секундах
    logger.info(f"Запуск планировщика. Интервал: {interval} секунд.")

    # Инициализация бота
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

    # parse_mode 'HTML' или 'Markdown' можете добавить при необходимости
    bot = TelegramBot(token=bot_token) if bot_token else None

    while True:
        # Выполняем наш "один цикл"
        if bot is not None:
            await run_sync_job(bot, max_client=max_client)
        elif max_client is not None:
            await run_sync_job(None, max_client=max_client)
        logger.info(f"Ожидание {interval} секунд до следующего запуска...")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Работа планировщика остановлена вручную.")
