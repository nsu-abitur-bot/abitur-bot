from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Topic


class TopicService:
    """Сервис для работы с темами сообщений."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_topic(
        self,
        label: str,
        description: Optional[str] = None,
        is_active: bool = True,
    ) -> Topic:
        """Создает новую тему."""
        topic = Topic(
            label=label,
            description=description,
            is_active=is_active,
        )
        self.session.add(topic)
        await self.session.commit()
        await self.session.refresh(topic)
        return topic

    async def get_topic_by_id(self, topic_id: int) -> Optional[Topic]:
        """Получает тему по ID."""
        stmt = select(Topic).where(Topic.id == topic_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_topic_by_label(self, label: str) -> Optional[Topic]:
        """Получает тему по названию."""
        stmt = select(Topic).where(Topic.label == label)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_active_topics(self) -> Sequence[Topic]:
        """Получает все активные темы."""
        stmt = select(Topic).where(Topic.is_active == True)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_topic(
        self,
        topic_id: int,
        label: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Topic]:
        """Обновляет тему."""
        topic = await self.get_topic_by_id(topic_id)
        if not topic:
            return None
        if label is not None:
            topic.label = label
        if description is not None:
            topic.description = description
        if is_active is not None:
            topic.is_active = is_active
        await self.session.commit()
        await self.session.refresh(topic)
        return topic

    async def delete_topic(self, topic_id: int) -> bool:
        """Удаляет тему (деактивирует)."""
        topic = await self.get_topic_by_id(topic_id)
        if not topic:
            return False
        topic.is_active = False
        await self.session.commit()
        return True