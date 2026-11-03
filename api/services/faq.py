from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.faq import FaqItem
from db.postgres.services.faq import FaqDbService

VERIFICATION_FIELDS = {
    "verification_status",
    "source_document",
    "source_location",
    "confirmed_by",
    "confirmed_at",
}


class FaqService:
    def __init__(self, session: AsyncSession):
        self._db = FaqDbService(session)

    async def get_all(self) -> list[FaqItem]:
        entries = await self._db.get_all()
        return [FaqItem.model_validate(e, from_attributes=True) for e in entries]

    async def create(self, item: FaqItem) -> FaqItem:
        entry = await self._db.create(
            item.question,
            item.aliases,
            item.answer,
            metadata=item.model_dump(include=VERIFICATION_FIELDS),
        )
        return FaqItem.model_validate(entry, from_attributes=True)

    async def create_many(self, items: list[FaqItem]) -> list[FaqItem]:
        raw = [i.model_dump(exclude={"id"}) for i in items]
        entries = await self._db.create_many(raw)
        return [FaqItem.model_validate(e, from_attributes=True) for e in entries]

    async def update(self, item_id: str, item: FaqItem) -> FaqItem:
        metadata = item.model_dump(include=VERIFICATION_FIELDS & item.model_fields_set)
        entry = await self._db.update(
            item_id, item.question, item.aliases, item.answer, metadata=metadata
        )
        if not entry:
            raise KeyError(item_id)
        return FaqItem.model_validate(entry, from_attributes=True)

    async def delete(self, item_id: str) -> None:
        deleted = await self._db.delete(item_id)
        if not deleted:
            raise KeyError(item_id)
