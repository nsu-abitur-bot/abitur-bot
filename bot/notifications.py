import asyncio
import enum
import logging
import os
from dataclasses import dataclass
from typing import Any, List

from aiogram import Bot

from db.postgres.db import AsyncSessionLocal
from db.postgres.dto import RatingChange
from db.postgres.models import User
from db.postgres.services.notification_outbox import NotificationOutboxService

logger = logging.getLogger(__name__)

# Пауза между отправками: очередь после долгого простоя может накопить сотни
# сообщений, а Telegram режет примерно на 30 в секунду. Полноценный троттлинг
# с обработкой flood wait — #292; здесь только чтобы не уйти в 429 залпом.
SEND_DELAY_SECONDS = float(os.getenv("NOTIFY_SEND_DELAY", "0.035"))
OUTBOX_BATCH_SIZE = 100


def format_notification_text(change: RatingChange) -> str | None:
    """Текст уведомления об одном изменении; None — сообщать нечего."""
    if change.is_new:
        return (
            f"🎉 Абитуриент {change.applicant_id} появился в рейтингах!\n\n"
            f"🔸 Направление: {change.direction} ({change.url})\n"
            f"🔹 Место: {change.new_place}\n"
            f"🔹 Статус: {change.new_status or '—'}\n"
            f"🔹 Конкурс: {change.new_competition_type or '—'}"
        )

    diff = []
    if change.old_place != change.new_place:
        diff.append(f"Место: {change.old_place} ➡️ {change.new_place}")
    if change.old_status != change.new_status:
        diff.append(f"Статус: {change.old_status or '—'} ➡️ {change.new_status or '—'}")
    if change.old_competition_type != change.new_competition_type:
        diff.append(
            f"Конкурс: {change.old_competition_type or '—'}"
            + f" ➡️ {change.new_competition_type or '—'}"
        )

    if not diff:
        return None
    return (
        f"🔔 Обновление позиции (Абитуриент {change.applicant_id})\n\n"
        f"🔸 Направление: {change.direction} ({change.url})\n"
    ) + "\n".join(f"🔸 {d}" for d in diff)


def generate_notification_texts(changes: List[RatingChange]) -> List[tuple[int, str]]:
    """Формирует список сообщений для уведомления пользователей."""
    notifications = []
    for change in changes:
        text = format_notification_text(change)
        if text:
            notifications.append((change.user_id, text))
    return notifications


class DeliveryStatus(enum.Enum):
    DELIVERED = "delivered"
    # Ни одного канала связи: повтор не поможет.
    NO_CHANNEL = "no_channel"
    # Каналы есть, но все отправки упали: стоит попробовать позже.
    FAILED = "failed"


@dataclass
class DeliveryResult:
    status: DeliveryStatus
    error: str | None = None


async def send_to_user(
    bot: Bot | None,
    user: User,
    text: str,
    max_client: Any | None = None,
) -> DeliveryResult:
    """Отправляет текст пользователю во все доступные каналы.

    Доставленным считается, если ушло хотя бы в один канал: повтор при
    частичном успехе продублировал бы сообщение там, где оно уже дошло.
    """
    attempted = False
    delivered = False
    errors: list[str] = []

    if bot and user.telegram_id:
        attempted = True
        try:
            await bot.send_message(chat_id=user.telegram_id, text=text)
            delivered = True
            logger.info(
                "Уведомление отправлено в Telegram: internal user_id=%s telegram_id=%s",
                user.user_id,
                user.telegram_id,
            )
        except Exception as e:
            errors.append(f"telegram: {e}")
            logger.error(
                "Ошибка отправки в Telegram пользователю %s: %s", user.user_id, e
            )

    if max_client and user.max_id:
        try:
            max_user_id = int(user.max_id)
        except ValueError:
            logger.warning(
                "Пропуск MAX уведомления: internal user_id=%s, некорректный max_id=%s",
                user.user_id,
                user.max_id,
            )
            max_user_id = None

        if max_user_id is not None:
            attempted = True
            try:
                await max_client.send_message(user_id=max_user_id, text=text)
                delivered = True
                logger.info(
                    "Уведомление отправлено в MAX: internal user_id=%s max_id=%s",
                    user.user_id,
                    user.max_id,
                )
            except Exception as e:
                errors.append(f"max: {e}")
                logger.error("Ошибка отправки в MAX пользователю %s: %s", user.user_id, e)

    if delivered:
        return DeliveryResult(DeliveryStatus.DELIVERED)
    if not attempted:
        logger.warning(
            "У пользователя %s нет доступных каналов для уведомления", user.user_id
        )
        return DeliveryResult(DeliveryStatus.NO_CHANNEL, "нет доступного канала связи")
    return DeliveryResult(DeliveryStatus.FAILED, "; ".join(errors))


async def send_pending_notifications(
    bot: Bot | None,
    max_client: Any | None = None,
    delay_seconds: float | None = None,
) -> int:
    """Рассылает всё неотправленное из notification_outbox.

    Каждое уведомление помечается сразу после отправки, отдельным коммитом:
    если процесс упадёт посреди рассылки, повторно уйдёт максимум одно
    сообщение, а не вся пачка. Упавшие отправки остаются в очереди до
    следующего прохода (с ограничением числа попыток).

    Возвращает число доставленных уведомлений.
    """
    delay = SEND_DELAY_SECONDS if delay_seconds is None else delay_seconds
    delivered = 0
    last_id = 0

    async with AsyncSessionLocal() as session:
        outbox = NotificationOutboxService(session)
        while True:
            # Курсор по id, а не «первые pending»: неудачные остаются pending,
            # и без курсора цикл крутился бы на них бесконечно.
            items = await outbox.get_pending(limit=OUTBOX_BATCH_SIZE, after_id=last_id)
            if not items:
                break
            for item in items:
                last_id = item.id
                text = format_notification_text(RatingChange(**item.payload))
                if not text:
                    await outbox.mark_undeliverable(item, "пустой текст уведомления")
                    continue
                if item.user is None:
                    await outbox.mark_undeliverable(item, "пользователь не найден")
                    continue

                result = await send_to_user(bot, item.user, text, max_client=max_client)
                if result.status is DeliveryStatus.DELIVERED:
                    await outbox.mark_sent(item)
                    delivered += 1
                elif result.status is DeliveryStatus.NO_CHANNEL:
                    await outbox.mark_undeliverable(item, result.error or "")
                else:
                    await outbox.mark_attempt_failed(item, result.error or "")

                if delay > 0:
                    await asyncio.sleep(delay)

    if delivered:
        logger.info("Отправлено уведомлений: %d", delivered)
    return delivered
