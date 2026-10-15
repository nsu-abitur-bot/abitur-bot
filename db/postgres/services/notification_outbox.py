import logging
from dataclasses import asdict
from typing import List

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..dto import RatingChange
from ..models import NotificationOutbox, NotificationStatus, timestamp

logger = logging.getLogger(__name__)

# После стольких неудачных попыток уведомление помечается failed и больше
# не отправляется: иначе один заблокировавший бота пользователь висел бы
# в очереди вечно.
MAX_DELIVERY_ATTEMPTS = 5


class NotificationOutboxService:
    """Очередь уведомлений об изменениях в конкурсных списках."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def enqueue(self, changes: List[RatingChange]) -> None:
        """Добавить уведомления в текущую транзакцию, не коммитя её.

        Коммит делает вызывающий — вместе с изменениями, ради которых
        уведомления появились.
        """
        self.session.add_all(
            NotificationOutbox(
                user_id=change.user_id,
                payload=asdict(change),
                status=NotificationStatus.pending.value,
                attempts=0,
            )
            for change in changes
        )

    async def get_pending(
        self, limit: int = 100, after_id: int = 0
    ) -> List[NotificationOutbox]:
        """Неотправленные уведомления по порядку постановки, вместе с адресатом."""
        result = await self.session.execute(
            select(NotificationOutbox)
            .options(selectinload(NotificationOutbox.user))
            .where(
                NotificationOutbox.status == NotificationStatus.pending.value,
                NotificationOutbox.id > after_id,
            )
            .order_by(NotificationOutbox.id)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_sent(self, item: NotificationOutbox) -> None:
        item.status = NotificationStatus.sent.value
        item.attempts += 1
        item.sent_at = timestamp()
        item.last_error = None
        await self.session.commit()

    async def mark_undeliverable(self, item: NotificationOutbox, reason: str) -> None:
        """Повторять бессмысленно: нет канала связи или адресата."""
        item.status = NotificationStatus.failed.value
        item.last_error = reason
        await self.session.commit()

    async def mark_attempt_failed(
        self,
        item: NotificationOutbox,
        error: str,
        max_attempts: int = MAX_DELIVERY_ATTEMPTS,
    ) -> None:
        """Попытка не удалась: оставить в очереди или сдаться после max_attempts."""
        item.attempts += 1
        item.last_error = error
        if item.attempts >= max_attempts:
            item.status = NotificationStatus.failed.value
            logger.error(
                "Уведомление %s не доставлено за %d попыток, больше не пробуем: %s",
                item.id,
                item.attempts,
                error,
            )
        await self.session.commit()

    async def count_pending(self) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(NotificationOutbox)
            .where(NotificationOutbox.status == NotificationStatus.pending.value)
        )
        return int(result.scalar_one())
