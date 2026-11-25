import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import MessageLog
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.settings import SettingsService

logger = logging.getLogger(__name__)

SYSTEM_RATE_LIMIT_MESSAGE = (
    "Сегодняшний общий лимит запросов к боту исчерпан. Попробуйте завтра."
)
USER_RATE_LIMIT_MESSAGE = "Вы исчерпали дневной лимит запросов к боту. Попробуйте завтра."


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    message: str | None = None


async def check_rate_limit(
    user_id: int, *, allow_on_database_error: bool = True
) -> RateLimitResult:
    day_start = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    try:
        async with AsyncSessionLocal() as session:
            settings = await SettingsService(session).get_rate_limit_settings()
            logs = MessageLogService(session)
            if await logs.count_user_inputs_since(day_start) >= (
                settings.system_requests_per_day
            ):
                return RateLimitResult(False, SYSTEM_RATE_LIMIT_MESSAGE)
            if await logs.count_user_inputs_since(day_start, user_id=user_id) >= (
                settings.user_requests_per_day
            ):
                return RateLimitResult(False, USER_RATE_LIMIT_MESSAGE)
    except Exception:
        if not allow_on_database_error:
            raise
        logger.exception("Ошибка проверки rate limit для user_id=%s", user_id)
    return RateLimitResult(True)


async def log_chat_question(
    user_id: int, session_id: str, message: str, source: str
) -> MessageLog:
    async with AsyncSessionLocal() as session:
        return await MessageLogService(session).create_log(
            user_id=user_id,
            session_id=session_id,
            message_type="user_input",
            content=message,
            message_metadata={"source": source},
        )
