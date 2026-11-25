import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.mini_app import MiniAppOwner, require_mini_app_owner
from config.chat import MAX_USER_MESSAGE_LENGTH
from db.postgres.db import get_async_session
from db.postgres.models import AnswerFeedback, MiniAppConversation, MiniAppTurn, timestamp
from db.postgres.services.answer_feedback import AnswerFeedbackService
from db.postgres.services.mini_app import MiniAppChatService, turn_status
from db.postgres.services.user import UserService
from db.redis.client import HISTORY_LIMIT
from pipeline.chat_access import check_rate_limit, log_chat_question
from pipeline.llm_client import LlmAnswer, ask_local_llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mini-app/chats", tags=["mini-app"])
Owner = Annotated[MiniAppOwner, Depends(require_mini_app_owner)]
Session = Annotated[AsyncSession, Depends(get_async_session)]
Rating = Literal["positive", "negative"]


class CreateChat(BaseModel):
    id: UUID


class Question(BaseModel):
    id: UUID
    text: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_USER_MESSAGE_LENGTH
        ),
    ]


class RateAnswer(BaseModel):
    rating: Rating


class ChatSummary(BaseModel):
    id: str
    title: str
    updated_at: datetime


class Source(BaseModel):
    title: str
    url: str


class Turn(BaseModel):
    id: str
    question: str
    status: str
    answer: str | None
    sources: list[Source]
    can_rate: bool
    rating: str | None


class Chat(ChatSummary):
    turns: list[Turn]


def _service(session: AsyncSession, owner: MiniAppOwner) -> MiniAppChatService:
    return MiniAppChatService(
        session, owner.instance_id, owner.platform, owner.external_id
    )


def _summary(chat: MiniAppConversation) -> ChatSummary:
    return ChatSummary(
        id=chat.id, title=chat.title, updated_at=chat.updated_at.replace(tzinfo=UTC)
    )


def _turn(turn: MiniAppTurn, rating: str | None = None) -> Turn:
    return Turn(
        id=turn.id,
        question=turn.question,
        status=turn_status(turn, timestamp()),
        answer=turn.answer,
        sources=[Source(**source) for source in turn.sources],
        can_rate=turn.feedback_id is not None,
        rating=rating,
    )


def _sources(answer: LlmAnswer) -> list[dict]:
    result = []
    seen = set()
    for source in answer.sources:
        url = str(source.get("url") or "").strip()
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.hostname or url in seen:
            continue
        seen.add(url)
        result.append(
            {"title": str(source.get("title") or "Источник информации"), "url": url}
        )
    return result


async def _owned_chat(
    service: MiniAppChatService, chat_id: UUID, *, lock: bool = False
) -> MiniAppConversation:
    chat = await service.get_chat(str(chat_id), lock=lock)
    if chat is None:
        raise HTTPException(404, "Диалог не найден")
    return chat


async def _user_id(session: AsyncSession, owner: MiniAppOwner) -> int:
    users = UserService(session)
    user = (
        await users.ensure_user_by_telegram_id(owner.external_id)
        if owner.platform == "telegram"
        else await users.ensure_user_by_max_id(str(owner.external_id))
    )
    return user.user_id


@router.get("")
async def list_chats(owner: Owner, session: Session) -> list[ChatSummary]:
    return [_summary(chat) for chat in await _service(session, owner).list_chats()]


@router.post("", status_code=201)
async def create_chat(body: CreateChat, owner: Owner, session: Session) -> ChatSummary:
    async with session.begin():
        chat = await _service(session, owner).create_chat(str(body.id))
        if chat is None:
            raise HTTPException(404, "Диалог не найден")
        return _summary(chat)


@router.get("/{chat_id}")
async def get_chat(chat_id: UUID, owner: Owner, session: Session) -> Chat:
    service = _service(session, owner)
    chat = await _owned_chat(service, chat_id)
    return Chat(
        **_summary(chat).model_dump(),
        turns=[_turn(turn, rating) for turn, rating in await service.list_turns(chat.id)],
    )


@router.post("/{chat_id}/turns")
async def send_question(
    chat_id: UUID, body: Question, owner: Owner, session: Session
) -> Turn:
    if "\x00" in body.text:
        raise HTTPException(422, "Некорректный текст сообщения")
    service = _service(session, owner)
    turn_id = str(body.id)
    async with session.begin():
        chat = await _owned_chat(service, chat_id, lock=True)
        turn = await service.get_turn(chat.id, turn_id)
        if turn is not None and turn.question != body.text:
            raise HTTPException(409, "Этот идентификатор уже занят другим сообщением")
        if turn is not None and turn.status == "complete":
            feedback = (
                await session.get(AnswerFeedback, turn.feedback_id)
                if (turn.feedback_id)
                else None
            )
            return _turn(turn, feedback.rating if feedback else None)
        if await service.active_turn(chat.id, timestamp()):
            raise HTTPException(409, "В этом диалоге уже готовится ответ")

        previous = await service.recent_turns(chat.id, HISTORY_LIMIT // 2)
        history = [
            entry
            for item in previous
            for entry in [{"role": "user", "content": item.question}, item.history_entry]
        ]
        attempt = service.begin_turn(chat, turn, turn_id, body.text).attempt

    session_id = f"mini-app:{chat_id}"
    answer: LlmAnswer | None = None
    user_id = None
    try:
        async with asyncio.timeout(85):
            user_id = await _user_id(session, owner)
            await session.commit()
            rate = await check_rate_limit(user_id, allow_on_database_error=False)
            if not rate.allowed:
                raise HTTPException(429, rate.message)
            log = await log_chat_question(user_id, session_id, body.text, "mini-app")
            answer = await ask_local_llm(
                body.text,
                session_id=session_id,
                user_id=user_id,
                log_entry_id=log.id,
                prior_history=history,
            )
    except HTTPException:
        await _finish_failed(session, service, chat_id, turn_id, attempt)
        raise
    except Exception:
        logger.exception("Ошибка обработки вопроса Mini App: chat_id=%s", chat_id)
        await session.rollback()

    if user_id is None or not answer or answer.failed:
        await _finish_failed(session, service, chat_id, turn_id, attempt)
        raise HTTPException(503, "Ответ не получен. Попробуйте отправить вопрос ещё раз.")

    async with session.begin():
        chat = await _owned_chat(service, chat_id, lock=True)
        turn = await service.get_turn(chat.id, turn_id)
        if (
            turn is None
            or turn.attempt != attempt
            or turn_status(turn, timestamp()) != "pending"
        ):
            raise HTTPException(409, "Повторная отправка уже началась")

        feedback = await AnswerFeedbackService(session).create_candidate(
            user_id=user_id,
            session_id=session_id,
            channel="mini-app",
            question=body.text,
            bot_response=answer.text,
            commit=False,
        )
        turn.status = "complete"
        turn.answer = str(answer.history_entry.get("content") or answer.text)
        turn.sources = _sources(answer)
        turn.history_entry = answer.history_entry or {
            "role": "assistant",
            "content": turn.answer,
        }
        turn.feedback_id = feedback.id
        if not previous:
            chat.title = " ".join(body.text.split())[:80]
        chat.updated_at = timestamp()
        return _turn(turn)


async def _finish_failed(
    session: AsyncSession,
    service: MiniAppChatService,
    chat_id: UUID,
    turn_id: str,
    attempt: int,
) -> None:
    async with session.begin():
        chat = await _owned_chat(service, chat_id, lock=True)
        turn = await service.get_turn(chat.id, turn_id)
        if turn is not None and turn.attempt == attempt:
            turn.status = "failed"


@router.put("/{chat_id}/turns/{turn_id}/rating")
async def rate_answer(
    chat_id: UUID, turn_id: UUID, body: RateAnswer, owner: Owner, session: Session
) -> dict[str, str]:
    service = _service(session, owner)
    chat = await _owned_chat(service, chat_id)
    turn = await service.get_turn(chat.id, str(turn_id))
    if turn is None or turn.feedback_id is None:
        raise HTTPException(404, "Ответ не найден")
    user_id = await _user_id(session, owner)
    feedback_id = turn.feedback_id
    if await AnswerFeedbackService(session).rate(feedback_id, user_id, body.rating):
        return {"rating": body.rating}
    feedback = await session.get(AnswerFeedback, feedback_id)
    if feedback is None or feedback.user_id != user_id or feedback.rating is None:
        raise HTTPException(404, "Ответ не найден")
    return {"rating": feedback.rating}
