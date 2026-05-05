from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.db import get_db
from db.postgres.services.topic import TopicService

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/", response_model=List[dict])
async def get_topics(db: AsyncSession = Depends(get_db)):
    """Получить список всех активных тем."""
    topic_service = TopicService(db)
    topics = await topic_service.get_all_active_topics()
    return [{"id": topic.id, "label": topic.label} for topic in topics]