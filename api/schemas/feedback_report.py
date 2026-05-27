from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

FeedbackStatus = Literal["open", "reviewed", "ignored"]


class FeedbackReportResponse(BaseModel):
    id: int
    user_id: int
    session_id: str
    channel: str
    comment: str
    question: Optional[str] = None
    bot_response: Optional[str] = None
    logs_snapshot: Optional[list[dict]] = None
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FeedbackReportListItem(BaseModel):
    id: int
    user_id: int
    session_id: str
    channel: str
    comment: str
    question: Optional[str] = None
    bot_response: Optional[str] = None
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FeedbackReportListResponse(BaseModel):
    reports: list[FeedbackReportListItem]
    total: int
    limit: int
    offset: int


class FeedbackReportQueryParams(BaseModel):
    status: Optional[FeedbackStatus] = Field(None, description="Статус обращения")
    user_id: Optional[int] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    limit: int = Field(50, ge=1, le=1000, description="Лимит записей")
    offset: int = Field(0, ge=0, description="Сдвиг")


class FeedbackReportStatusUpdate(BaseModel):
    status: FeedbackStatus
