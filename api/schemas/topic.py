from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TopicBase(BaseModel):
    """Базовая модель темы."""
    label: str = Field(..., min_length=1, max_length=255, description="Название темы")
    description: Optional[str] = Field(None, description="Описание темы")


class TopicCreate(TopicBase):
    """Модель для создания новой темы."""
    is_active: bool = Field(True, description="Активна ли тема")


class TopicUpdate(BaseModel):
    """Модель для обновления темы."""
    label: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Название темы"
    )
    description: Optional[str] = Field(None, description="Описание темы")
    is_active: Optional[bool] = Field(None, description="Активна ли тема")


class TopicResponse(TopicBase):
    """Модель ответа для темы."""
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TopicListResponse(BaseModel):
    """Модель ответа для списка тем."""
    topics: List[TopicResponse]
    total: int
