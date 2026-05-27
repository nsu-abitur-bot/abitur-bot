from sqlalchemy.ext.asyncio import AsyncSession

from abbrev.expander import get_abbrev_expander
from api.schemas.abbrev import AbbrevItem
from db.postgres.services.abbrev import AbbrevDbService


class AbbrevService:
    def __init__(self, session: AsyncSession):
        self._db = AbbrevDbService(session)

    async def get_all(self) -> list[AbbrevItem]:
        entries = await self._db.get_all()
        return [AbbrevItem(id=e.id, short=e.short, full=e.full) for e in entries]

    async def create(self, item: AbbrevItem) -> AbbrevItem:
        entry = await self._db.create(item.short, item.full)
        await self._reload_expander()
        return AbbrevItem(id=entry.id, short=entry.short, full=entry.full)

    async def update(self, item_id: str, item: AbbrevItem) -> AbbrevItem:
        entry = await self._db.update(item_id, item.short, item.full)
        if not entry:
            raise KeyError(item_id)
        await self._reload_expander()
        return AbbrevItem(id=entry.id, short=entry.short, full=entry.full)

    async def delete(self, item_id: str) -> None:
        deleted = await self._db.delete(item_id)
        if not deleted:
            raise KeyError(item_id)
        await self._reload_expander()

    async def _reload_expander(self) -> None:
        entries = await self._db.get_all()
        items = [{"short": e.short, "full": e.full} for e in entries]
        get_abbrev_expander().load_items(items)
