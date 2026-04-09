import logging
from typing import Any, List

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from db.postgres.db import AsyncSessionLocal
from db.postgres.dto import RatingChange
from db.postgres.services.user import UserService

logger = logging.getLogger(__name__)


def generate_notification_texts(changes: List[RatingChange]) -> List[tuple[int, str]]:
    """Формирует список сообщений для уведомления пользователей.

    Анализирует старые и новые данные (RatingChange) и собирает текст.
    """
    notifications = []

    for change in changes:
        if change.is_new:
            text = (
                f"🎉 Абитуриент {change.applicant_id} появился в рейтингах!\n\n"
                f"🔸 Направление: {change.direction} ({change.url})\n"
                f"🔹 Место: {change.new_place}\n"
                f"🔹 Статус: {change.new_status or '—'}\n"
                f"🔹 Конкурс: {change.new_competition_type or '—'}"
            )
            notifications.append((change.user_id, text))
            continue

        diff = []
        if change.old_place != change.new_place:
            diff.append(f"Место: {change.old_place} ➡️ {change.new_place}")
        if change.old_status != change.new_status:
            diff.append(
                f"Статус: {change.old_status or '—'} ➡️ {change.new_status or '—'}"
            )
        if change.old_competition_type != change.new_competition_type:
            diff.append(
                f"Конкурс: {change.old_competition_type or '—'}"
                + f" ➡️ {change.new_competition_type or '—'}"
            )

        if diff:
            text = (
                f"🔔 Обновление позиции (Абитуриент {change.applicant_id})\n\n"
                f"🔸 Направление: {change.direction} ({change.url})\n"
            ) + "\n".join(f"🔸 {d}" for d in diff)
            notifications.append((change.user_id, text))

    return notifications


async def notify_users(
    bot: Bot | None,
    changes: List[RatingChange],
    max_client: Any | None = None,
) -> None:
    """Определяет, кого нужно уведомить, и рассылает сообщения по каналам."""
    notifications = generate_notification_texts(changes)

    if not notifications:
        logger.info("Нет изменений для отправки уведомлений.")
        return

    for user_id, text in notifications:
        try:
            async with AsyncSessionLocal() as session:
                user_service = UserService(session)
                user = await user_service.get_user(user_id)

            if not user:
                logger.warning(
                    "Пропуск уведомления: internal user_id=%s не найден",
                    user_id,
                )
                continue

            if bot and user.telegram_id:
                await bot.send_message(chat_id=user.telegram_id, text=text)
                logger.info(
                    (
                        "Уведомление отправлено в Telegram: "
                        "internal user_id=%s telegram_id=%s"
                    ),
                    user_id,
                    user.telegram_id,
                )

            if max_client and user.max_id:
                try:
                    max_user_id = int(user.max_id)
                except ValueError:
                    logger.warning(
                        "Пропуск MAX уведомления: internal user_id=%s, "
                        "некорректный max_id=%s",
                        user_id,
                        user.max_id,
                    )
                    max_user_id = None

                if max_user_id is not None:
                    await max_client.send_message(user_id=max_user_id, text=text)
                    logger.info(
                        "Уведомление отправлено в MAX: internal user_id=%s max_id=%s",
                        user_id,
                        user.max_id,
                    )

            if (not bot or not user.telegram_id) and (
                not max_client or not user.max_id
            ):
                logger.warning(
                    "У пользователя %s нет доступных каналов для уведомления",
                    user_id,
                )
        except TelegramAPIError as e:
            logger.error("Ошибка при отправке пользователю %s: %s", user_id, e)
        except Exception as e:
            logger.error("Неизвестная ошибка при отправке %s: %s", user_id, e)
