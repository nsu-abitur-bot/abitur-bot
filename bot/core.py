import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Awaitable, Callable, Optional

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import MessageLog
from db.postgres.services.feedback_report import FeedbackReportService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.settings import SettingsService
from db.postgres.services.user import UserService
from db.redis.client import get_redis_client
from llm.llm_client import ask_local_llm

StreamCallback = Callable[[str], Awaitable[None]]
StatusCallback = Callable[[str], Awaitable[None]]

logger = logging.getLogger(__name__)


WELCOME_TEXT = (
    "Привет! 👋 Я — умный ИИ-помощник для абитуриентов. "
    "Моя главная цель — сберечь ваши нервы и сделать процесс "
    "поступления проще и понятнее.\n\n"
    "<b>Что я умею:</b>\n"
    "🎓 <b>Отвечать на вопросы</b>\n"
    "Просто напишите мне любой вопрос про поступление, проходные баллы, "
    "документы или саму учебу, и я постараюсь дать максимально точный "
    "и подробный ответ.\n\n"
    "📊 <b>Следить за конкурсными списками</b>\n"
    "Используйте эти команды:\n"
    "• /track — добавьте свой идентификационный номер (или СНИЛС), "
    "и я буду присылать уведомление каждый раз, когда ваша позиция "
    "в списке будет меняться.\n"
    "• /untrack — удалить номер из отслеживания.\n"
    "• /reset — очистить историю переписки.\n"
    "• /feedback — поделиться обратной связью, если я ответил неправильно.\n"
    "• /cancel — отменить ввод обратной связи или другое ожидаемое действие.\n\n"
    "💡 <i>Небольшой факт: этот бот — проект студентов второго курса. "
    "Мы сами не так давно проходили через все этапы поступления, "
    "поэтому решили создать инструмент, которого нам самим тогда "
    "очень не хватало!</i>\n\n"
    "Напишите свой вопрос или используйте команду /track, чтобы начать! 🚀"
)

FEEDBACK_FOOTER = (
    "Я ответил неправильно?\n"
    "Поделитесь обратной связью командой /feedback"
)

SYSTEM_RATE_LIMIT_MESSAGE = (
    "Сегодняшний общий лимит запросов к боту исчерпан. Попробуйте завтра."
)
USER_RATE_LIMIT_MESSAGE = (
    "Вы исчерпали дневной лимит запросов к боту. Попробуйте завтра."
)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    message: str | None = None


class BotReply:
    def __init__(
        self,
        text: str,
        parse_mode: str | None = None,
        fallback_plain_on_format_error: bool = False,
        include_feedback_footer: bool = False,
    ):
        self.text = self._with_feedback_footer(text) if include_feedback_footer else text
        self.parse_mode = parse_mode
        self.fallback_plain_on_format_error = fallback_plain_on_format_error

    @staticmethod
    def _with_feedback_footer(text: str) -> str:
        if not text or FEEDBACK_FOOTER in text:
            return text
        return f"{text}\n\n{FEEDBACK_FOOTER}"


class BotCore:
    """Общая бизнес-логика бота, независимая от транспорта."""

    async def resolve_internal_user_id(
        self, channel: str, external_user_id: str
    ) -> int:
        async with AsyncSessionLocal() as session:
            user_service = UserService(session)
            if channel == "telegram":
                user = await user_service.ensure_user_by_telegram_id(
                    int(external_user_id)
                )
            elif channel == "max":
                user = await user_service.ensure_user_by_max_id(external_user_id)
            else:
                raise ValueError(f"Unsupported channel: {channel}")
        return user.user_id

    async def _save_user_message_to_db(
        self, user_id: int, session_id: str, message: str, source: str
    ) -> Optional[MessageLog]:
        try:
            async with AsyncSessionLocal() as db_session:
                log_service = MessageLogService(db_session)
                log_entry = await log_service.create_log(
                    user_id=user_id,
                    session_id=session_id,
                    message_type="user_input",
                    content=message,
                    message_metadata={"source": source},
                )
                return log_entry
        except Exception as e:
            logger.error(f"Ошибка сохранения входящего сообщения в БД: {e}")
            return None

    async def _save_feedback_report(
        self,
        user_id: int,
        session_id: str,
        channel: str,
        comment: str,
    ) -> None:
        async with AsyncSessionLocal() as db_session:
            report_service = FeedbackReportService(db_session)
            await report_service.create_report(
                user_id=user_id,
                session_id=session_id,
                channel=channel,
                comment=comment,
            )

    async def _check_rate_limit(self, user_id: int) -> RateLimitResult:
        day_start = datetime.now(UTC).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None,
        )
        try:
            async with AsyncSessionLocal() as session:
                settings = await SettingsService(session).get_rate_limit_settings()
                log_service = MessageLogService(session)
                system_count = await log_service.count_user_inputs_since(day_start)
                if system_count >= settings.system_requests_per_day:
                    return RateLimitResult(
                        allowed=False,
                        message=SYSTEM_RATE_LIMIT_MESSAGE,
                    )

                user_count = await log_service.count_user_inputs_since(
                    day_start,
                    user_id=user_id,
                )
                if user_count >= settings.user_requests_per_day:
                    return RateLimitResult(
                        allowed=False,
                        message=USER_RATE_LIMIT_MESSAGE,
                    )
        except Exception:
            logger.exception("Ошибка проверки rate limit для user_id=%s", user_id)
            return RateLimitResult(allowed=True)

        return RateLimitResult(allowed=True)

    async def cmd_start(
        self, channel: str, external_user_id: str, session_id: str
    ) -> BotReply:
        internal_user_id = await self.resolve_internal_user_id(
            channel, external_user_id
        )
        redis_client = await get_redis_client()
        await redis_client.set_awaiting_applicant_id(session_id, False)
        await redis_client.set_awaiting_feedback(session_id, False)
        logger.info(
            "Команда /start: channel=%s external_user_id=%s internal_user_id=%s",
            channel,
            external_user_id,
            internal_user_id,
        )
        return BotReply(
            text=WELCOME_TEXT,
            parse_mode="HTML",
            fallback_plain_on_format_error=False,
        )

    async def cmd_track(self, session_id: str) -> BotReply:
        redis_client = await get_redis_client()
        await redis_client.set_awaiting_feedback(session_id, False)
        await redis_client.set_awaiting_applicant_id(session_id, True)
        return BotReply(text="Укажите свой идентификатор абитуриента")

    async def cmd_untrack(
        self, channel: str, external_user_id: str, session_id: str
    ) -> BotReply:
        internal_user_id = await self.resolve_internal_user_id(
            channel, external_user_id
        )
        async with AsyncSessionLocal() as session:
            user_service = UserService(session)
            updated = await user_service.update_applicant_id(internal_user_id, None)

        redis_client = await get_redis_client()
        await redis_client.set_awaiting_applicant_id(session_id, False)
        await redis_client.set_awaiting_feedback(session_id, False)

        if not updated:
            logger.warning(
                (
                    "Не удалось выполнить /untrack: channel=%s "
                    "external_user_id=%s internal_user_id=%s"
                ),
                channel,
                external_user_id,
                internal_user_id,
            )
            return BotReply(text="Отслеживание уже было отключено")

        return BotReply(text="Отслеживание прекращено")

    async def cmd_reset(self, session_id: str) -> BotReply:
        redis_client = await get_redis_client()
        new_session_id = await redis_client.reset_dialog_session(session_id)
        await redis_client.set_awaiting_applicant_id(session_id, False)
        await redis_client.set_awaiting_feedback(session_id, False)
        logger.info(
            "Команда /reset: base_session_id=%s new_session_id=%s",
            session_id,
            new_session_id,
        )
        return BotReply(text="История переписки очищена")

    async def cmd_feedback(self, session_id: str) -> BotReply:
        redis_client = await get_redis_client()
        await redis_client.set_awaiting_applicant_id(session_id, False)
        await redis_client.set_awaiting_feedback(session_id, True)
        return BotReply(
            text=(
                "Напишите одним сообщением, что бот сделал не так. "
                "Если передумали, отправьте /cancel."
            ),
            include_feedback_footer=False,
        )

    async def cmd_cancel(self, session_id: str) -> BotReply:
        redis_client = await get_redis_client()
        was_awaiting_feedback = await redis_client.is_awaiting_feedback(session_id)
        was_awaiting_applicant_id = await redis_client.is_awaiting_applicant_id(
            session_id
        )
        await redis_client.set_awaiting_feedback(session_id, False)
        await redis_client.set_awaiting_applicant_id(session_id, False)

        if was_awaiting_feedback:
            text = "Отменил отправку обратной связи."
        elif was_awaiting_applicant_id:
            text = "Отменил текущее действие."
        else:
            text = "Сейчас нечего отменять."

        return BotReply(text=text, include_feedback_footer=False)

    async def handle_message(
        self,
        channel: str,
        external_user_id: str,
        session_id: str,
        user_name: str,
        user_text: str,
        stream_callback: Optional[StreamCallback] = None,
        status_callback: Optional[StatusCallback] = None,
    ) -> BotReply:
        internal_user_id = await self.resolve_internal_user_id(
            channel, external_user_id
        )
        redis_client = await get_redis_client()
        dialog_session_id = await redis_client.get_dialog_session_id(session_id)

        logger.info(
            (
                "Сообщение: channel=%s external_user_id=%s "
                "base_session_id=%s dialog_session_id=%s text=%s"
            ),
            channel,
            external_user_id,
            session_id,
            dialog_session_id,
            user_text,
        )

        if await redis_client.is_awaiting_feedback(session_id):
            if not user_text.strip():
                return BotReply(
                    text=(
                        "Сообщение не может быть пустым. Опишите, что пошло не так, "
                        "или отправьте /cancel."
                    ),
                    include_feedback_footer=False,
                )

            await self._save_feedback_report(
                user_id=internal_user_id,
                session_id=dialog_session_id,
                channel=channel,
                comment=user_text,
            )
            await redis_client.set_awaiting_feedback(session_id, False)
            return BotReply(
                text="Спасибо, мы получили обратную связь и разберем этот ответ.",
                include_feedback_footer=False,
            )

        is_awaiting = await redis_client.is_awaiting_applicant_id(session_id)

        if is_awaiting:
            if len(user_text) > 7:
                return BotReply(
                    text=(
                        "Идентификатор должен быть не длиннее 7 символов. "
                        "Попробуйте еще раз."
                    )
                )

            if not user_text.strip():
                return BotReply(
                    text="Идентификатор не может быть пустым. Попробуйте еще раз."
                )

            async with AsyncSessionLocal() as session:
                user_service = UserService(session)
                updated = await user_service.update_applicant_id(
                    internal_user_id, user_text
                )

            if updated:
                await redis_client.set_awaiting_applicant_id(session_id, False)
                return BotReply(
                    text=(
                        f"Отслеживание абитуриента {user_text} началось! "
                        "Я пришлю уведомление при изменении позиции."
                    )
                )

            return BotReply(
                text=(
                    "Ошибка при сохранении идентификатора. "
                    "Проверьте формат и попробуйте снова."
                )
            )

        rate_limit = await self._check_rate_limit(internal_user_id)
        if not rate_limit.allowed:
            return BotReply(
                text=rate_limit.message or USER_RATE_LIMIT_MESSAGE,
                include_feedback_footer=False,
            )

        formatted_message = f"[from {user_name}] {user_text}"

        log_entry = await self._save_user_message_to_db(
            user_id=internal_user_id,
            session_id=dialog_session_id,
            message=formatted_message,
            source=channel,
        )

        response = await ask_local_llm(
            formatted_message,
            session_id=dialog_session_id,
            user_id=internal_user_id,
            log_entry_id=log_entry.id if log_entry else None,
            stream_callback=stream_callback,
            status_callback=status_callback,
        )

        if not response:
            return BotReply(text="Ответ не найден")

        return BotReply(
            text=response,
            parse_mode="HTML" if channel == "telegram" else None,
            fallback_plain_on_format_error=(channel == "telegram"),
            include_feedback_footer=True,
        )
