from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.faq import router as faq_router
from api.routes.rag import router as rag_router
from db.postgres.db import get_session
from db.postgres.services.message import MessageService
from db.postgres.services.user import UserService


class UserCountStatsResponse(BaseModel):
    day: int
    week: int
    month: int
    year: int
    all_time: int


class MessageResponse(BaseModel):
    id: str
    user_id: int
    session_id: str
    user_text: str
    bot_response: str
    created_at: datetime


app = FastAPI(title="Abitur API")

# Регистрируем роутер FAQ
app.include_router(faq_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")


@app.get("/api/v1/users/count-stats", response_model=UserCountStatsResponse)
async def get_users_count_stats(
    session: AsyncSession = Depends(get_session),
) -> UserCountStatsResponse:
    service = UserService(session)
    try:
        stats = await service.get_user_count_stats()
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return UserCountStatsResponse(**stats)


@app.get("/api/v1/messages", response_model=List[MessageResponse])
async def get_messages(
    user_id: Optional[int] = Query(None, description="Фильтр по user_id"),
    limit: int = Query(50, ge=1, le=500, description="Количество записей"),
    offset: int = Query(0, ge=0, description="Смещение"),
    session: AsyncSession = Depends(get_session),
) -> List[MessageResponse]:
    """Возвращает список сообщений и ответов бота."""
    service = MessageService(session)
    try:
        if user_id is not None:
            messages = await service.get_messages_by_user(
                user_id=user_id, limit=limit, offset=offset
            )
        else:
            messages = await service.get_all_messages(limit=limit, offset=offset)
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return [
        MessageResponse(
            id=msg.id,
            user_id=msg.user_id,
            session_id=msg.session_id,
            user_text=msg.user_text,
            bot_response=msg.bot_response,
            created_at=msg.created_at,
        )
        for msg in messages
    ]
