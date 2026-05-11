import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Abbreviation

logger = logging.getLogger(__name__)


class AbbrevDbService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> list[Abbreviation]:
        try:
            result = await self.session.execute(
                select(Abbreviation).order_by(Abbreviation.created_at)
            )
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Ошибка получения всех аббревиатур: {e}")
            return []

    async def create(self, short: str, full: str) -> Abbreviation:
        entry = Abbreviation(short=short, full=full)
        self.session.add(entry)
        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            raise ValueError(f"Аббревиатура '{short}' уже существует")
        await self.session.refresh(entry)
        return entry

    async def get(self, item_id: str) -> Optional[Abbreviation]:
        result = await self.session.execute(
            select(Abbreviation).where(Abbreviation.id == item_id)
        )
        return result.scalar_one_or_none()

    async def update(self, item_id: str, short: str, full: str) -> Optional[Abbreviation]:
        entry = await self.get(item_id)
        if not entry:
            return None
        entry.short = short
        entry.full = full
        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            raise ValueError(f"Аббревиатура '{short}' уже существует")
        await self.session.refresh(entry)
        return entry

    async def delete(self, item_id: str) -> bool:
        entry = await self.get(item_id)
        if not entry:
            return False
        await self.session.delete(entry)
        await self.session.commit()
        return True
