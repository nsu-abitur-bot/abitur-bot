import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.message_log import MessageLogService


@pytest.mark.asyncio
async def test_get_logs_by_base_session_includes_dialog_sessions(
    session: AsyncSession,
):
    service = MessageLogService(session)

    await service.create_log(
        user_id=1,
        session_id="session-1",
        message_type="user_input",
        content="Старый вопрос",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_query",
        content="Запрос к базе знаний",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_response",
        content="Ответ базы знаний",
    )
    await service.create_log(
        user_id=1,
        session_id="session-2",
        message_type="user_input",
        content="Чужой вопрос",
    )

    logs = await service.get_logs_by_session("session-1")

    assert {log.session_id for log in logs} == {"session-1", "session-1:dialog:1"}
    assert {log.message_type for log in logs} == {
        "user_input",
        "rag_query",
        "rag_response",
    }


@pytest.mark.asyncio
async def test_get_logs_by_dialog_session_returns_only_that_dialog(
    session: AsyncSession,
):
    service = MessageLogService(session)

    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:1",
        message_type="rag_response",
        content="Первый диалог",
    )
    await service.create_log(
        user_id=1,
        session_id="session-1:dialog:2",
        message_type="rag_response",
        content="Второй диалог",
    )

    logs = await service.get_logs_by_session("session-1:dialog:1")

    assert len(logs) == 1
    assert logs[0].content == "Первый диалог"
