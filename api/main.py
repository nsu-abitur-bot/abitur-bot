from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.db import get_session
from db.postgres.services.user import UserService


class UserCountStatsResponse(BaseModel):
    day: int
    week: int
    month: int
    year: int
    all_time: int


app = FastAPI(title="Abitur API")


@app.get("/api/v1/users/count", response_model=UserCountStatsResponse)
async def get_users_count(
    session: AsyncSession = Depends(get_session),
) -> UserCountStatsResponse:
    service = UserService(session)
    stats = await service.get_user_count_stats()
    return UserCountStatsResponse(**stats)
