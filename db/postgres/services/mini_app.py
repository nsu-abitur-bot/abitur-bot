from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AnswerFeedback, MiniAppConversation, MiniAppTurn, timestamp

LEASE_DURATION = timedelta(seconds=105)


def turn_status(turn: MiniAppTurn, now: datetime) -> str:
    if turn.status == "pending" and now - turn.started_at >= LEASE_DURATION:
        return "failed"
    return turn.status


class MiniAppChatService:
    def __init__(
        self, session: AsyncSession, instance_id: str, platform: str, external_id: int
    ) -> None:
        self.session = session
        self.owner = dict(
            instance_id=instance_id, platform=platform, external_id=external_id
        )

    async def list_chats(self) -> list[MiniAppConversation]:
        stmt = (
            select(MiniAppConversation)
            .filter_by(**self.owner)
            .order_by(
                MiniAppConversation.updated_at.desc(), MiniAppConversation.id.desc()
            )
        )
        return list(await self.session.scalars(stmt))

    async def get_chat(
        self, chat_id: str, *, lock: bool = False
    ) -> MiniAppConversation | None:
        stmt = (
            select(MiniAppConversation)
            .filter_by(id=chat_id, **self.owner)
            .execution_options(populate_existing=True)
        )
        if lock:
            stmt = stmt.with_for_update()
        return await self.session.scalar(stmt)

    async def create_chat(self, chat_id: str) -> MiniAppConversation | None:
        await self.session.execute(
            insert(MiniAppConversation)
            .values(id=chat_id, **self.owner)
            .on_conflict_do_nothing(index_elements=["id"])
        )
        return await self.get_chat(chat_id)

    async def list_turns(self, chat_id: str) -> list[tuple[MiniAppTurn, str | None]]:
        stmt = (
            select(MiniAppTurn, AnswerFeedback.rating)
            .outerjoin(AnswerFeedback, MiniAppTurn.feedback_id == AnswerFeedback.id)
            .where(MiniAppTurn.conversation_id == chat_id)
            .order_by(MiniAppTurn.started_at, MiniAppTurn.id)
        )
        return [
            (turn, rating) for turn, rating in (await self.session.execute(stmt)).all()
        ]

    async def active_turn(self, chat_id: str, now: datetime) -> str | None:
        return await self.session.scalar(
            select(MiniAppTurn.id)
            .filter_by(conversation_id=chat_id, status="pending")
            .where(MiniAppTurn.started_at > now - LEASE_DURATION)
        )

    async def get_turn(self, chat_id: str, turn_id: str) -> MiniAppTurn | None:
        return await self.session.get(
            MiniAppTurn, (chat_id, turn_id), populate_existing=True
        )

    async def recent_turns(self, chat_id: str, limit: int) -> list[MiniAppTurn]:
        stmt = (
            select(MiniAppTurn)
            .filter_by(conversation_id=chat_id, status="complete")
            .order_by(MiniAppTurn.started_at.desc(), MiniAppTurn.id.desc())
            .limit(limit)
        )
        return list(reversed(list(await self.session.scalars(stmt))))

    def begin_turn(
        self, chat: MiniAppConversation, turn: MiniAppTurn | None, turn_id: str, text: str
    ) -> MiniAppTurn:
        now = timestamp()
        if turn is None:
            turn = MiniAppTurn(
                id=turn_id,
                conversation_id=chat.id,
                question=text,
                attempt=0,
            )
            self.session.add(turn)
        turn.attempt += 1
        turn.started_at = now
        turn.status = "pending"
        chat.updated_at = now
        return turn
