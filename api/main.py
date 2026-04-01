import os
import re
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.faq import router as faq_router
from api.routes.message_log import router as message_log_router
from api.routes.rag import router as rag_router
from db.postgres.db import get_async_session
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
    username: Optional[str] = None
    session_id: str
    user_text: str
    bot_response: str
    created_at: datetime


app = FastAPI(title="Abitur API")

cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
cors_origins = [
    origin.strip() for origin in cors_origins_env.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрируем роутеры
app.include_router(faq_router, prefix="/api/v1")
app.include_router(message_log_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")


@app.get("/api/v1/users/count-stats", response_model=UserCountStatsResponse)
async def get_users_count_stats(
    session: AsyncSession = Depends(get_async_session),
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
    session: AsyncSession = Depends(get_async_session),
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

    result = []
    for msg in messages:
        # Извлекаем username из формата [from username] text
        username = None
        user_text = msg.user_text
        match = re.match(r"^\[from (.*?)\] (.*)$", user_text, re.DOTALL)
        if match:
            username = match.group(1)
            user_text = match.group(2)

        result.append(
            MessageResponse(
                id=msg.id,
                user_id=msg.user_id,
                username=username,
                session_id=msg.session_id,
                user_text=user_text,  # Возвращаем основной текст без префикса
                bot_response=msg.bot_response,
                created_at=msg.created_at,
            )
        )
    return result
