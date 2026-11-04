import logging
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import FaqEntry

logger = logging.getLogger(__name__)
FAQ_CHANGE_CHANNEL = "abitur_faq_changed"


class FaqDbService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> list[FaqEntry]:
        try:
            result = await self.session.execute(
                select(FaqEntry).order_by(FaqEntry.created_at)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Ошибка получения всех FAQ: {e}")
            return []

    async def create(
        self,
        question: str,
        aliases: list[str],
        answer: str,
        metadata: dict | None = None,
    ) -> FaqEntry:
        entry = FaqEntry(
            question=question, aliases=aliases, answer=answer, **(metadata or {})
        )
        self.session.add(entry)
        try:
            await self._notify_changed()
            await self.session.commit()
            await self.session.refresh(entry)
            return entry
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания FAQ: {e}")
            raise

    async def create_many(self, items: list[dict]) -> list[FaqEntry]:
        entries = [
            FaqEntry(
                question=item["question"],
                aliases=item.get("aliases", []),
                answer=item["answer"],
                verification_status=item.get("verification_status", "draft"),
                source_document=item.get("source_document"),
                source_location=item.get("source_location"),
                confirmed_by=item.get("confirmed_by"),
                confirmed_at=item.get("confirmed_at"),
            )
            for item in items
        ]
        self.session.add_all(entries)
        try:
            await self._notify_changed()
            await self.session.commit()
            for entry in entries:
                await self.session.refresh(entry)
        except Exception:
            await self.session.rollback()
            logger.exception("Ошибка массового создания FAQ")
            raise
        return entries

    async def get(self, item_id: str) -> Optional[FaqEntry]:
        result = await self.session.execute(
            select(FaqEntry).where(FaqEntry.id == item_id)
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        item_id: str,
        question: str,
        aliases: list[str],
        answer: str,
        metadata: dict | None = None,
    ) -> Optional[FaqEntry]:
        entry = await self.get(item_id)
        if not entry:
            return None
        entry.question = question
        entry.aliases = aliases
        entry.answer = answer
        for key, value in (metadata or {}).items():
            setattr(entry, key, value)
        try:
            await self._notify_changed()
            await self.session.commit()
            await self.session.refresh(entry)
        except Exception:
            await self.session.rollback()
            raise
        return entry

    async def delete(self, item_id: str) -> bool:
        entry = await self.get(item_id)
        if not entry:
            return False
        await self.session.delete(entry)
        try:
            await self._notify_changed()
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return True

    async def _notify_changed(self) -> None:
        await self.session.execute(
            text("SELECT pg_notify(:channel, '')"), {"channel": FAQ_CHANGE_CHANNEL}
        )
