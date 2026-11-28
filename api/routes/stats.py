from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.db import get_db
from db.postgres.models import Leaderboard, MessageLog, Topic

router = APIRouter(prefix="/stats", tags=["stats"])


class ParsingStatsResponse(BaseModel):
    last_parsed_at: str | None = Field(
        default=None,
        description=(
            "Время последнего успешного парсинга или проверки активных списков (UTC)"
        ),
    )
    data_updated_at: str | None = Field(
        default=None,
        description=(
            "Время последнего фактического изменения данных активных списков (UTC)"
        ),
    )
    total_lists: int = Field(
        default=0,
        description="Количество активных рейтинговых списков",
    )


@router.get("/parsing", response_model=ParsingStatsResponse)
async def get_parsing_stats(db: AsyncSession = Depends(get_db)):
    """Получить статистику парсинга рейтинговых списков.

    `last_parsed_at` отражает время последней успешной проверки/парсинга
    (включая неизменившиеся списки), а `data_updated_at` — время последнего
    обновления данных при изменении содержимого. `updated_at` записи не
    используется, чтобы правки метаданных не подменяли дату парсинга.
    """
    stmt = select(
        func.max(Leaderboard.last_parsed_at).label("last_parsed_at"),
        func.max(Leaderboard.data_updated_at).label("data_updated_at"),
        func.count(Leaderboard.id).label("total_lists"),
    ).where(Leaderboard.is_active)
    result = await db.execute(stmt)
    row = result.one_or_none()
    last_parsed_at = row.last_parsed_at if row else None
    data_updated_at = getattr(row, "data_updated_at", None) if row else None
    total_lists = row.total_lists if row else 0
    return ParsingStatsResponse(
        last_parsed_at=last_parsed_at.isoformat() if last_parsed_at else None,
        data_updated_at=data_updated_at.isoformat() if data_updated_at else None,
        total_lists=total_lists,
    )


@router.get("/topics", response_model=List[dict])
async def get_topic_stats(db: AsyncSession = Depends(get_db)):
    """Получить статистику по темам сообщений."""
    stmt = (
        select(Topic.label, func.count(MessageLog.id).label("count"))
        .join(MessageLog, Topic.id == MessageLog.topic_id, isouter=True)
        .where(Topic.is_active)
        .group_by(Topic.id, Topic.label)
        .order_by(func.count(MessageLog.id).desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [{"topic": row.label, "count": row.count} for row in rows]
