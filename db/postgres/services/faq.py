import logging
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import FaqEntry, normalize_question, timestamp

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
        except IntegrityError:
            await self.session.rollback()
            raise ValueError(f"FAQ с вопросом '{question}' уже существует")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания FAQ: {e}")
            raise

    async def upsert_many(self, items: list[dict]) -> tuple[list[FaqEntry], int, int]:
        """Массовый импорт FAQ с UPSERT по нормализованному вопросу (#295).

        Повторная загрузка файла обновляет существующие записи, а не дублирует их.
        Верификационные поля (verification_status и provenance) не сбрасываются.
        Возвращает (записи, сколько создано, сколько обновлено).
        """
        # Дедуп внутри файла: побеждает последнее значение.
        deduped: dict[str, dict] = {}
        for item in items:
            deduped[normalize_question(item["question"])] = item
        if not deduped:
            return [], 0, 0

        keys = list(deduped.keys())
        result = await self.session.execute(
            select(FaqEntry.normalized_question).where(
                FaqEntry.normalized_question.in_(keys)
            )
        )
        existing_keys = set(result.scalars())
        updated = len(existing_keys)
        created = len(deduped) - updated

        values = [
            {
                "question": item["question"],
                "normalized_question": key,
                "aliases": item.get("aliases", []),
                "answer": item["answer"],
                "verification_status": item.get("verification_status", "draft"),
                "source_document": item.get("source_document"),
                "source_location": item.get("source_location"),
                "confirmed_by": item.get("confirmed_by"),
                "confirmed_at": item.get("confirmed_at"),
            }
            for key, item in deduped.items()
        ]
        stmt = pg_insert(FaqEntry).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[FaqEntry.normalized_question],
            set_={
                "question": stmt.excluded.question,
                "aliases": stmt.excluded.aliases,
                "answer": stmt.excluded.answer,
                "updated_at": timestamp(),
            },
        )
        try:
            await self.session.execute(stmt)
            await self._notify_changed()
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            logger.exception("Ошибка массового импорта FAQ")
            raise

        result = await self.session.execute(
            select(FaqEntry)
            .where(FaqEntry.normalized_question.in_(keys))
            .order_by(FaqEntry.created_at)
            .execution_options(populate_existing=True)
        )
        return list(result.scalars().all()), created, updated

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
        except IntegrityError:
            await self.session.rollback()
            raise ValueError(f"FAQ с вопросом '{question}' уже существует")
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
