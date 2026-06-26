import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Abbreviation, timestamp

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
        except Exception:
            await self.session.rollback()
            raise
        await self.session.refresh(entry)
        return entry

    async def upsert_many(self, items: list[dict]) -> list[Abbreviation]:
        """
        Массовый импорт аббревиатур с UPSERT по полю ``short``.

        Если ``short`` уже существует — обновляется ``full``.
        При повторе ``short`` внутри одного файла побеждает последнее
        значение (last-wins).
        """
        # Дедупликация внутри файла: побеждает последнее значение.
        deduped: dict[str, str] = {}
        for item in items:
            short = item["short"]
            full = item["full"]
            deduped[short] = full

        if not deduped:
            return []

        values = [{"short": short, "full": full} for short, full in deduped.items()]

        stmt = pg_insert(Abbreviation).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Abbreviation.short],
            set_={
                "full": stmt.excluded.full,
                "updated_at": timestamp(),
            },
        )
        try:
            await self.session.execute(stmt)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            logger.exception("Ошибка массового импорта аббревиатур")
            raise

        result = await self.session.execute(
            select(Abbreviation).where(Abbreviation.short.in_(deduped.keys()))
        )
        return list(result.scalars().all())

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
        except Exception:
            await self.session.rollback()
            raise
        await self.session.refresh(entry)
        return entry

    async def delete(self, item_id: str) -> bool:
        entry = await self.get(item_id)
        if not entry:
            return False
        await self.session.delete(entry)
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return True
