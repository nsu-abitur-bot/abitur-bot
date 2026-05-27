import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.feedback_report import FeedbackReportService
from db.postgres.services.message import MessageService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_create_feedback_report_with_context(session: AsyncSession):
    user_service = UserService(session)
    await user_service.create_user(user_id=200001)

    message_service = MessageService(session)
    await message_service.create_message(
        user_id=200001,
        session_id="feedback-session",
        user_text="[from user] Когда подать документы?",
        bot_response="До 25 июля.",
    )

    log_service = MessageLogService(session)
    await log_service.create_log(
        user_id=200001,
        session_id="feedback-session",
        message_type="llm_response",
        content="До 25 июля.",
        message_metadata={"provider": "test"},
    )

    service = FeedbackReportService(session)
    report = await service.create_report(
        user_id=200001,
        session_id="feedback-session",
        channel="telegram",
        comment="Срок был указан неверно",
    )

    assert report.id is not None
    assert report.status == "open"
    assert report.question == "[from user] Когда подать документы?"
    assert report.bot_response == "До 25 июля."
    assert report.logs_snapshot is not None
    assert report.logs_snapshot[0]["message_type"] == "llm_response"


@pytest.mark.asyncio
async def test_list_and_update_feedback_reports(session: AsyncSession):
    service = FeedbackReportService(session)
    first = await service.create_report(
        user_id=200002,
        session_id="session-1",
        channel="telegram",
        comment="Первый комментарий",
    )
    await service.create_report(
        user_id=200003,
        session_id="session-2",
        channel="max",
        comment="Второй комментарий",
    )

    reports, total = await service.list_reports(user_id=200002)
    assert total == 1
    assert reports[0].id == first.id

    updated = await service.update_status(first.id, "reviewed")
    assert updated is not None
    assert updated.status == "reviewed"
    assert updated.reviewed_at is not None

    reviewed, reviewed_total = await service.list_reports(status="reviewed")
    assert reviewed_total == 1
    assert reviewed[0].id == first.id


@pytest.mark.asyncio
async def test_update_feedback_report_invalid_status(session: AsyncSession):
    service = FeedbackReportService(session)

    with pytest.raises(ValueError):
        await service.update_status(1, "closed")
