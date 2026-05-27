from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import FeedbackReport, Message, MessageLog, timestamp

FEEDBACK_STATUSES = {"open", "reviewed", "ignored"}


class FeedbackReportService:
    """Сервис для работы с обратной связью пользователей."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_report(
        self,
        user_id: int,
        session_id: str,
        channel: str,
        comment: str,
        logs_limit: int = 50,
    ) -> FeedbackReport:
        latest_message = await self._get_latest_message(session_id)
        logs_snapshot = await self._get_logs_snapshot(session_id, logs_limit)

        report = FeedbackReport(
            user_id=user_id,
            session_id=session_id,
            channel=channel,
            comment=comment,
            question=latest_message.user_text if latest_message else None,
            bot_response=latest_message.bot_response if latest_message else None,
            logs_snapshot=logs_snapshot,
            status="open",
        )
        self.session.add(report)
        await self.session.commit()
        await self.session.refresh(report)
        return report

    async def get_report(self, report_id: int) -> Optional[FeedbackReport]:
        result = await self.session.execute(
            select(FeedbackReport).where(FeedbackReport.id == report_id)
        )
        return result.scalar_one_or_none()

    async def list_reports(
        self,
        status: Optional[str] = None,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FeedbackReport], int]:
        stmt = select(FeedbackReport)
        count_stmt = select(func.count(FeedbackReport.id))

        filters = []
        if status is not None:
            filters.append(FeedbackReport.status == status)
        if user_id is not None:
            filters.append(FeedbackReport.user_id == user_id)
        if session_id is not None:
            filters.append(FeedbackReport.session_id == session_id)

        if filters:
            stmt = stmt.where(*filters)
            count_stmt = count_stmt.where(*filters)

        stmt = (
            stmt.order_by(FeedbackReport.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        total = (await self.session.execute(count_stmt)).scalar_one()
        return list(rows), int(total)

    async def update_status(
        self, report_id: int, status: str
    ) -> Optional[FeedbackReport]:
        if status not in FEEDBACK_STATUSES:
            raise ValueError("Invalid feedback status")

        report = await self.get_report(report_id)
        if report is None:
            return None

        report.status = status
        report.reviewed_at = timestamp() if status in {"reviewed", "ignored"} else None
        await self.session.commit()
        await self.session.refresh(report)
        return report

    async def _get_latest_message(self, session_id: str) -> Optional[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_logs_snapshot(
        self, session_id: str, limit: int
    ) -> list[dict[str, object]]:
        result = await self.session.execute(
            select(MessageLog)
            .where(MessageLog.session_id == session_id)
            .order_by(MessageLog.created_at.desc())
            .limit(limit)
        )
        logs = result.scalars().all()
        return [
            {
                "id": log.id,
                "user_id": log.user_id,
                "session_id": log.session_id,
                "message_type": log.message_type,
                "content": log.content,
                "message_metadata": log.message_metadata,
                "topic_id": log.topic_id,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]
