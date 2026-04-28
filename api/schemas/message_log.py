from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MessageLogResponse(BaseModel):
    """Модель ответа для лога сообщения."""
    id: int
    user_id: int
    session_id: str
    message_type: str
    content: str
    message_metadata: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MessageLogListResponse(BaseModel):
    """Модель ответа для списка логов."""
    logs: List[MessageLogResponse]
    total: int
    limit: int
    offset: int


class MessageLogQueryParams(BaseModel):
    """Параметры запроса для получения логов."""
    user_id: Optional[int] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    message_type: Optional[str] = Field(None, description="Тип сообщения")
    limit: int = Field(50, ge=1, le=1000, description="Лимит записей")
    offset: int = Field(0, ge=0, description="Сдвиг")


class RequestCountBucket(BaseModel):
    """Элемент статистики по количеству запросов."""
    period: datetime
    count: int


class RequestCountStatsResponse(BaseModel):
    """Статистика количества запросов за период времени."""
    total: int
    group_by: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    buckets: List[RequestCountBucket]
