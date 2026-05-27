import logging
from typing import Optional

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import MessageLog
from db.postgres.services.feedback_report import FeedbackReportService
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.user import UserService
from db.redis.client import get_redis_client
from llm.llm_client import ask_local_llm

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


class BotReply:
    def __init__(
        self,
        text: str,
        parse_mode: str | None = None,
        fallback_plain_on_format_error: bool = False,
        include_feedback_footer: bool = True,
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
        await redis_client.clear_history(session_id)
        await redis_client.set_awaiting_applicant_id(session_id, False)
        await redis_client.set_awaiting_feedback(session_id, False)
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
    ) -> BotReply:
        internal_user_id = await self.resolve_internal_user_id(
            channel, external_user_id
        )
        redis_client = await get_redis_client()

        logger.info(
            "Сообщение: channel=%s external_user_id=%s session_id=%s text=%s",
            channel,
            external_user_id,
            session_id,
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
                session_id=session_id,
                channel=channel,
                comment=user_text,
            )
            await redis_client.set_awaiting_feedback(session_id, False)
            return BotReply(
                text="Спасибо, мы получили обратную связь и разберем этот ответ.",
                include_feedback_footer=False,
            )

        formatted_message = f"[from {user_name}] {user_text}"

        log_entry = await self._save_user_message_to_db(
            user_id=internal_user_id,
            session_id=session_id,
            message=formatted_message,
            source=channel,
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

        response = await ask_local_llm(
            formatted_message,
            session_id=session_id,
            user_id=internal_user_id,
            log_entry_id=log_entry.id if log_entry else None,
        )

        if not response:
            return BotReply(text="Ответ не найден")

        return BotReply(
            text=response,
            parse_mode="HTML" if channel == "telegram" else None,
            fallback_plain_on_format_error=(channel == "telegram"),
        )
