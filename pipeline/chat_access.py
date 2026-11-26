from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.message_log import MessageLogService
from db.postgres.services.settings import SettingsService


class RateLimitExceeded(StrEnum):
    SYSTEM = "Сегодняшний общий лимит запросов к боту исчерпан. Попробуйте завтра."
    USER = "Вы исчерпали дневной лимит запросов к боту. Попробуйте завтра."


async def reserve_chat_question(
    session: AsyncSession, user_id: int, session_id: str, message: str, source: str
) -> int | RateLimitExceeded:
    """Резервирует запрос в транзакции, которую вызывающий завершает до генерации."""
    # Один ключ для всех каналов и пользователей, поскольку системный лимит общий.
    await session.execute(text("SELECT pg_advisory_xact_lock(491, 1)"))
    day_start = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    settings = await SettingsService(session).get_rate_limit_settings()
    logs = MessageLogService(session)
    if await logs.count_user_inputs_since(day_start) >= settings.system_requests_per_day:
        return RateLimitExceeded.SYSTEM
    if await logs.count_user_inputs_since(day_start, user_id=user_id) >= (
        settings.user_requests_per_day
    ):
        return RateLimitExceeded.USER

    log = await logs.create_log(
        user_id=user_id,
        session_id=session_id,
        message_type="user_input",
        content=message,
        message_metadata={"source": source},
        commit=False,
    )
    return log.id
