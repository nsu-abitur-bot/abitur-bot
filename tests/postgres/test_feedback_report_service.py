import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.answer_feedback import AnswerFeedbackService
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
async def test_answer_rating_rolls_back_if_report_cannot_be_saved(session: AsyncSession):
    service = AnswerFeedbackService(session)
    candidate = await service.create_candidate(
        user_id=200001,
        session_id="dialog-1",
        channel="telegram",
        question="Когда подать документы?",
        bot_response="До 25 июля.",
    )
    feedback_id = candidate.id
    await session.execute(
        text(
            "ALTER TABLE feedback_report ADD CONSTRAINT reject_rating_report "
            "CHECK (comment <> '👎 Неверно')"
        )
    )
    await session.commit()

    with pytest.raises(IntegrityError):
        await service.rate(feedback_id, 200001, "negative")
    await session.rollback()
    await session.refresh(candidate)
    assert candidate.rating is None
    assert candidate.rated_at is None
    assert (await FeedbackReportService(session).list_reports())[1] == 0

    await session.execute(
        text("ALTER TABLE feedback_report DROP CONSTRAINT reject_rating_report")
    )
    await session.commit()
    assert await service.rate(feedback_id, 200001, "negative") is True


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


@pytest.mark.asyncio
async def test_answer_rating_is_saved_once_for_its_owner(session: AsyncSession):
    service = AnswerFeedbackService(session)
    candidate = await service.create_candidate(
        user_id=200001,
        session_id="dialog-1",
        channel="telegram",
        question="Когда подать документы?",
        bot_response="До 25 июля.",
    )

    assert await service.rate(candidate.id, 200002, "positive") is False
    assert await service.rate(candidate.id, 200001, "negative") is True
    assert await service.rate(candidate.id, 200001, "positive") is False

    await session.refresh(candidate)
    assert candidate.rating == "negative"
    assert candidate.rated_at is not None
    assert candidate.question == "Когда подать документы?"
    assert candidate.bot_response == "До 25 июля."

    reports, total = await FeedbackReportService(session).list_reports()
    assert total == 1
    assert reports[0].comment == "👎 Неверно"


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["telegram", "max"])
@pytest.mark.parametrize(
    ("rating", "comment"),
    [("positive", "👍 Полезно"), ("negative", "👎 Неверно")],
)
async def test_answer_rating_creates_report_for_rated_answer(
    session: AsyncSession, channel: str, rating: str, comment: str
):
    service = AnswerFeedbackService(session)
    candidate = await service.create_candidate(
        user_id=200001,
        session_id="dialog-1",
        channel=channel,
        question="Когда подать документы?",
        bot_response="До 25 июля.",
    )
    await UserService(session).create_user(user_id=200001)
    await MessageService(session).create_message(
        user_id=200001,
        session_id="dialog-1",
        user_text="А где общежитие?",
        bot_response="В Академгородке.",
    )
    reports = FeedbackReportService(session)
    assert (await reports.list_reports())[1] == 0
    assert await service.rate(candidate.id, 200002, rating) is False
    assert await service.rate("missing", 200001, rating) is False
    assert (await reports.list_reports())[1] == 0

    assert await service.rate(candidate.id, 200001, rating) is True
    assert await service.rate(candidate.id, 200001, rating) is False

    items, total = await reports.list_reports()
    assert total == 1
    report = items[0]
    assert report.user_id == candidate.user_id
    assert report.session_id == candidate.session_id
    assert report.channel == channel
    assert report.comment == comment
    assert report.question == candidate.question
    assert report.bot_response == candidate.bot_response
    assert report.status == "open"
