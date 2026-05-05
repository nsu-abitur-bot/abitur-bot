from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.db import get_db
from db.postgres.models import MessageLog, Topic

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/topics", response_model=List[dict])
async def get_topic_stats(db: AsyncSession = Depends(get_db)):
    """Получить статистику по темам сообщений."""
    stmt = (
        select(Topic.label, func.count(MessageLog.id).label("count"))
        .join(MessageLog, Topic.id == MessageLog.topic_id, isouter=True)
        .where(Topic.is_active == True)
        .group_by(Topic.id, Topic.label)
        .order_by(func.count(MessageLog.id).desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [{"topic": row.label, "count": row.count} for row in rows]