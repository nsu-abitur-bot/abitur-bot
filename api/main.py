from fastapi import Depends, FastAPI, HTTPException
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
