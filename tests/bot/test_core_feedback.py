from unittest.mock import AsyncMock

import pytest

from bot.core import FEEDBACK_FOOTER, BotCore


class FakeRedis:
    def __init__(self):
        self.awaiting_feedback = False
        self.awaiting_applicant_id = False
        self.history_cleared = False

    async def set_awaiting_feedback(self, session_id: str, value: bool) -> None:
        self.awaiting_feedback = value

    async def is_awaiting_feedback(self, session_id: str) -> bool:
        return self.awaiting_feedback

    async def set_awaiting_applicant_id(self, session_id: str, value: bool) -> None:
        self.awaiting_applicant_id = value

    async def is_awaiting_applicant_id(self, session_id: str) -> bool:
        return self.awaiting_applicant_id

    async def clear_history(self, session_id: str) -> None:
        self.history_cleared = True


@pytest.mark.asyncio
async def test_cmd_feedback_enters_feedback_mode(monkeypatch):
    redis = FakeRedis()

    async def fake_get_redis_client():
        return redis

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)

    reply = await BotCore().cmd_feedback("session-1")

    assert redis.awaiting_feedback
    assert "/cancel" in reply.text
    assert FEEDBACK_FOOTER not in reply.text


@pytest.mark.asyncio
async def test_feedback_message_creates_report_and_skips_llm(monkeypatch):
    redis = FakeRedis()
    redis.awaiting_feedback = True
    core = BotCore()
    saved_feedback = AsyncMock()
    ask_llm = AsyncMock(return_value="LLM response")

    async def fake_get_redis_client():
        return redis

    async def fake_resolve_internal_user_id(channel: str, external_user_id: str) -> int:
        return 42

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(core, "resolve_internal_user_id", fake_resolve_internal_user_id)
    monkeypatch.setattr(core, "_save_feedback_report", saved_feedback)
    monkeypatch.setattr("bot.core.ask_local_llm", ask_llm)

    reply = await core.handle_message(
        channel="telegram",
        external_user_id="123",
        session_id="session-1",
        user_name="user",
        user_text="Бот перепутал дату",
    )

    saved_feedback.assert_awaited_once_with(
        user_id=42,
        session_id="session-1",
        channel="telegram",
        comment="Бот перепутал дату",
    )
    ask_llm.assert_not_awaited()
    assert not redis.awaiting_feedback
    assert "Спасибо" in reply.text
    assert FEEDBACK_FOOTER not in reply.text


@pytest.mark.asyncio
async def test_cancel_exits_feedback_mode(monkeypatch):
    redis = FakeRedis()
    redis.awaiting_feedback = True

    async def fake_get_redis_client():
        return redis

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)

    reply = await BotCore().cmd_cancel("session-1")

    assert not redis.awaiting_feedback
    assert "Отменил отправку обратной связи" in reply.text
    assert FEEDBACK_FOOTER not in reply.text


@pytest.mark.asyncio
async def test_empty_feedback_message_keeps_feedback_mode(monkeypatch):
    redis = FakeRedis()
    redis.awaiting_feedback = True
    core = BotCore()
    saved_feedback = AsyncMock()

    async def fake_get_redis_client():
        return redis

    async def fake_resolve_internal_user_id(channel: str, external_user_id: str) -> int:
        return 42

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(core, "resolve_internal_user_id", fake_resolve_internal_user_id)
    monkeypatch.setattr(core, "_save_feedback_report", saved_feedback)

    reply = await core.handle_message(
        channel="telegram",
        external_user_id="123",
        session_id="session-1",
        user_name="user",
        user_text="",
    )

    saved_feedback.assert_not_awaited()
    assert redis.awaiting_feedback
    assert "Сообщение не может быть пустым" in reply.text
    assert FEEDBACK_FOOTER not in reply.text


@pytest.mark.asyncio
async def test_normal_reply_has_feedback_footer(monkeypatch):
    redis = FakeRedis()
    core = BotCore()
    user_log = type("Log", (), {"id": 10})()

    async def fake_get_redis_client():
        return redis

    async def fake_resolve_internal_user_id(channel: str, external_user_id: str) -> int:
        return 42

    async def fake_save_user_message_to_db(
        user_id: int, session_id: str, message: str, source: str
    ):
        return user_log

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(core, "resolve_internal_user_id", fake_resolve_internal_user_id)
    monkeypatch.setattr(core, "_save_user_message_to_db", fake_save_user_message_to_db)
    monkeypatch.setattr("bot.core.ask_local_llm", AsyncMock(return_value="Ответ бота"))

    reply = await core.handle_message(
        channel="telegram",
        external_user_id="123",
        session_id="session-1",
        user_name="user",
        user_text="Как поступить?",
    )

    assert reply.text.endswith(FEEDBACK_FOOTER)


@pytest.mark.asyncio
async def test_report_is_not_feedback_command(monkeypatch):
    redis = FakeRedis()
    core = BotCore()
    user_log = type("Log", (), {"id": 10})()
    ask_llm = AsyncMock(return_value="Ответ на report")

    async def fake_get_redis_client():
        return redis

    async def fake_resolve_internal_user_id(channel: str, external_user_id: str) -> int:
        return 42

    async def fake_save_user_message_to_db(
        user_id: int, session_id: str, message: str, source: str
    ):
        return user_log

    monkeypatch.setattr("bot.core.get_redis_client", fake_get_redis_client)
    monkeypatch.setattr(core, "resolve_internal_user_id", fake_resolve_internal_user_id)
    monkeypatch.setattr(core, "_save_user_message_to_db", fake_save_user_message_to_db)
    monkeypatch.setattr("bot.core.ask_local_llm", ask_llm)

    await core.handle_message(
        channel="telegram",
        external_user_id="123",
        session_id="session-1",
        user_name="user",
        user_text="/report",
    )

    ask_llm.assert_awaited_once()
    assert not redis.awaiting_feedback
