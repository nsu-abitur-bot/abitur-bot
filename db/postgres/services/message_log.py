from datetime import datetime
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.models import MessageLog


class MessageLogService:
    """Сервис для работы с логами сообщений."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_log(
        self,
        user_id: int,
        session_id: str,
        message_type: str,
        content: str,
        message_metadata: Optional[dict] = None,
    ) -> MessageLog:
        """Создает запись в логе сообщений."""
        log_entry = MessageLog(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            message_metadata=message_metadata,
        )
        self.session.add(log_entry)
        await self.session.commit()
        await self.session.refresh(log_entry)
        return log_entry

    async def get_logs_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает логи для конкретной сессии."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.session_id == session_id)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_logs_by_user(
        self,
        user_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает логи для конкретного пользователя."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.user_id == user_id)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_logs_by_type(
        self,
        message_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает логи по типу сообщения."""
        stmt = (
            select(MessageLog)
            .where(MessageLog.message_type == message_type)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_recent_logs(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MessageLog]:
        """Получает самые последние логи."""
        stmt = (
            select(MessageLog)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_request_count_stats(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        group_by: str,
        message_type: str = "user_input",
    ) -> dict:
        """Возвращает статистику количества запросов по периодам."""
        base_stmt = select(func.count(MessageLog.id)).where(
            MessageLog.message_type == message_type
        )

        if start is not None:
            base_stmt = base_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            base_stmt = base_stmt.where(MessageLog.created_at <= end)

        total = (await self.session.execute(base_stmt)).scalar_one()

        period_expr = func.date_trunc(group_by, MessageLog.created_at).label("period")
        bucket_stmt = select(period_expr, func.count(MessageLog.id).label("count")).where(
            MessageLog.message_type == message_type
        )
        if start is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at >= start)
        if end is not None:
            bucket_stmt = bucket_stmt.where(MessageLog.created_at <= end)

        bucket_stmt = bucket_stmt.group_by(period_expr).order_by(period_expr)
        rows = (await self.session.execute(bucket_stmt)).all()

        buckets = [
            {"period": row.period, "count": row.count}
            for row in rows
            if row.period is not None
        ]

        return {"total": total, "buckets": buckets}
