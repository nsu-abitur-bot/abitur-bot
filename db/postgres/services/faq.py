import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import FaqEntry

logger = logging.getLogger(__name__)


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
        self, question: str, aliases: list[str], answer: str
    ) -> FaqEntry:
        entry = FaqEntry(question=question, aliases=aliases, answer=answer)
        self.session.add(entry)
        try:
            await self.session.commit()
            await self.session.refresh(entry)
            return entry
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания FAQ: {e}")
            raise

    async def create_many(
        self, items: list[dict]
    ) -> list[FaqEntry]:
        entries = [
            FaqEntry(
                question=item["question"],
                aliases=item.get("aliases", []),
                answer=item["answer"],
            )
            for item in items
        ]
        self.session.add_all(entries)
        await self.session.commit()
        for entry in entries:
            await self.session.refresh(entry)
        return entries

    async def get(self, item_id: str) -> Optional[FaqEntry]:
        result = await self.session.execute(
            select(FaqEntry).where(FaqEntry.id == item_id)
        )
        return result.scalar_one_or_none()

    async def update(
        self, item_id: str, question: str, aliases: list[str], answer: str
    ) -> Optional[FaqEntry]:
        entry = await self.get(item_id)
        if not entry:
            return None
        entry.question = question
        entry.aliases = aliases
        entry.answer = answer
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def delete(self, item_id: str) -> bool:
        entry = await self.get(item_id)
        if not entry:
            return False
        await self.session.delete(entry)
        await self.session.commit()
        return True
