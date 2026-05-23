import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from abbrev.expander import get_abbrev_expander
from api.auth.dependencies import get_current_admin, require_admin
from api.auth.router import router as auth_router
from api.routes.abbrev import router as abbrev_router
from api.routes.evals import router as evals_router
from api.routes.faq import router as faq_router
from api.routes.message_log import router as message_log_router
from api.routes.rag import router as rag_router
from api.routes.stats import router as stats_router
from api.routes.system_logs import router as system_logs_router
from api.routes.topic import router as topic_router
from db.postgres.db import AsyncSessionLocal, get_async_session
from db.postgres.models import Admin
from db.postgres.services.abbrev import AbbrevDbService
from db.postgres.services.faq import FaqDbService
from db.postgres.services.message import MessageService
from db.postgres.services.user import UserService
from faq.matcher import get_faq_matcher

logger = logging.getLogger(__name__)


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
    messenger: Optional[str] = None
    session_id: str
    user_text: str
    bot_response: str
    created_at: datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSessionLocal() as session:
        try:
            faq_entries = await FaqDbService(session).get_all()
            get_faq_matcher().load_items(
                [
                    {"question": e.question, "aliases": e.aliases, "answer": e.answer}
                    for e in faq_entries
                ]
            )
            logger.info("FAQ загружен из БД: %d записей", len(faq_entries))

            abbrev_entries = await AbbrevDbService(session).get_all()
            get_abbrev_expander().load_items(
                [{"short": e.short, "full": e.full} for e in abbrev_entries]
            )
            logger.info("Аббревиатуры загружены из БД: %d записей", len(abbrev_entries))
        except Exception:
            logger.exception("Ошибка загрузки FAQ/аббревиатур из БД при старте")
    yield


app = FastAPI(title="Abitur API", lifespan=lifespan)

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

# Auth роутер — публичные эндпоинты /login и /register, остальное защищено внутри
app.include_router(auth_router, prefix="/api/v1")

# Публичные роуты
app.include_router(topic_router, prefix="/api/v1")

# Все остальные роутеры требуют аутентификации
# Мутации (abbrev, faq, rag, evals) требуют роль admin или superadmin
# Аналитика (message_log, stats) доступна любому авторизованному, включая viewer
app.include_router(
    abbrev_router,
    prefix="/api/v1",
    dependencies=[Depends(require_admin)],
)
app.include_router(
    faq_router,
    prefix="/api/v1",
    dependencies=[Depends(require_admin)],
)
app.include_router(
    message_log_router,
    prefix="/api/v1",
    dependencies=[Depends(get_current_admin)],
)
app.include_router(
    rag_router,
    prefix="/api/v1",
    dependencies=[Depends(require_admin)],
)
app.include_router(
    evals_router,
    prefix="/api/v1",
    dependencies=[Depends(require_admin)],
)
app.include_router(
    stats_router,
    prefix="/api/v1",
    dependencies=[Depends(get_current_admin)],
)
app.include_router(
    system_logs_router,
    prefix="/api/v1",
    dependencies=[Depends(require_admin)],
)


@app.get("/api/v1/users/count-stats", response_model=UserCountStatsResponse)
async def get_users_count_stats(
    session: AsyncSession = Depends(get_async_session),
    _: Admin = Depends(get_current_admin),
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
    _: Admin = Depends(get_current_admin),
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

        if getattr(msg, "user", None):
            if msg.user.telegram_id is not None:
                messenger = "Telegram"
            elif msg.user.max_id is not None:
                messenger = "MAX"
            else:
                messenger = None
        else:
            messenger = None

        result.append(
            MessageResponse(
                id=msg.id,
                user_id=msg.user_id,
                username=username,
                messenger=messenger,
                session_id=msg.session_id,
                user_text=user_text,
                bot_response=msg.bot_response,
                created_at=msg.created_at,
            )
        )
    return result
