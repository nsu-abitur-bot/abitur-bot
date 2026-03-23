import logging
from typing import List

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from db.postgres.dto import RatingChange

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


async def notify_users(bot: Bot, changes: List[RatingChange]) -> None:
    """Определяет, кого нужно уведомить, генерирует текст и рассылает сообщения."""
    notifications = generate_notification_texts(changes)

    if not notifications:
        logger.info("Нет изменений для отправки уведомлений.")
        return

    for user_id, text in notifications:
        try:
            await bot.send_message(chat_id=user_id, text=text)
            logger.info("Уведомление отправлено пользователю %s", user_id)
        except TelegramAPIError as e:
            logger.error("Ошибка при отправке пользователю %s: %s", user_id, e)
        except Exception as e:
            logger.error("Неизвестная ошибка при отправке %s: %s", user_id, e)
