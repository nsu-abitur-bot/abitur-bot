from datetime import datetime, timedelta
from typing import List, Optional, TypedDict

from sqlalchemy import Sequence, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.postgres.models import MessageLog


def _truncate_dt(dt: datetime, group_by: str) -> datetime:
    if group_by == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if group_by == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if group_by == "week":
        return (dt - timedelta(days=dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if group_by == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt


def _advance_dt(dt: datetime, group_by: str) -> datetime:
    if group_by == "hour":
        return dt + timedelta(hours=1)
    if group_by == "day":
        return dt + timedelta(days=1)
    if group_by == "week":
        return dt + timedelta(weeks=1)
    if group_by == "month":
        month = dt.month % 12 + 1
        year = dt.year + (1 if dt.month == 12 else 0)
        return dt.replace(year=year, month=month)
    return dt


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
    ) -> Sequence[MessageLog]:
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
    ) -> Sequence[MessageLog]:
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
    ) -> Sequence[MessageLog]:
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
    ) -> Sequence[MessageLog]:
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
        if start is not None:
            start = start.replace(tzinfo=None)
        if end is not None:
            end = end.replace(tzinfo=None)

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

        valid_rows = [row for row in rows if row.period is not None]
        if valid_rows:
            range_start = (
                _truncate_dt(start, group_by)
                if start is not None
                else min(row.period for row in valid_rows)
            )
            range_end = end if end is not None else max(row.period for row in valid_rows)
            all_buckets: dict[datetime, int] = {}
            current = range_start
            while current <= range_end:
                all_buckets[current] = 0
                current = _advance_dt(current, group_by)
            for row in valid_rows:
                if row.period in all_buckets:
                    count_value = row._mapping["count"]
                    all_buckets[row.period] = int(count_value)
            buckets = [{"period": k, "count": v} for k, v in sorted(all_buckets.items())]
        else:
            buckets = []

        return {"total": total, "buckets": buckets}

    class PopularQuestionRow(TypedDict):
        question: str
        count: int

    async def get_popular_questions(
        self,
        limit: int = 10,
    ) -> List[PopularQuestionRow]:
        """Возвращает самые часто встречающиеся вопросы пользователей."""
        stmt = (
            select(
                MessageLog.content.label("question"),
                func.count(MessageLog.id).label("count"),
            )
            .where(MessageLog.message_type == "user_input")
            .group_by(MessageLog.content)
            .order_by(func.count(MessageLog.id).desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "question": str(row._mapping["question"]),
                "count": int(row._mapping["count"]),
            }
            for row in rows
            if row._mapping["question"] is not None
        ]
