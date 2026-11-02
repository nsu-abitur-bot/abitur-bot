from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.faq import FaqItem
from db.postgres.services.faq import FaqDbService


class FaqService:
    def __init__(self, session: AsyncSession):
        self._db = FaqDbService(session)

    async def get_all(self) -> list[FaqItem]:
        entries = await self._db.get_all()
        return [
            FaqItem(id=e.id, question=e.question, aliases=e.aliases, answer=e.answer)
            for e in entries
        ]

    async def create(self, item: FaqItem) -> FaqItem:
        entry = await self._db.create(item.question, item.aliases, item.answer)
        return FaqItem(
            id=entry.id,
            question=entry.question,
            aliases=entry.aliases,
            answer=entry.answer,
        )

    async def create_many(self, items: list[FaqItem]) -> list[FaqItem]:
        raw = [
            {"question": i.question, "aliases": i.aliases, "answer": i.answer}
            for i in items
        ]
        entries = await self._db.create_many(raw)
        return [
            FaqItem(id=e.id, question=e.question, aliases=e.aliases, answer=e.answer)
            for e in entries
        ]

    async def update(self, item_id: str, item: FaqItem) -> FaqItem:
        entry = await self._db.update(item_id, item.question, item.aliases, item.answer)
        if not entry:
            raise KeyError(item_id)
        return FaqItem(
            id=entry.id,
            question=entry.question,
            aliases=entry.aliases,
            answer=entry.answer,
        )

    async def delete(self, item_id: str) -> None:
        deleted = await self._db.delete(item_id)
        if not deleted:
            raise KeyError(item_id)
