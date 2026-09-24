from typing import Optional

from pydantic import BaseModel, Field


class FaqItem(BaseModel):
    id: Optional[str] = Field(None, description="UUID записи (заполняется при ответе)")
    question: str = Field(..., description="Основной вопрос")
    aliases: list[str] = Field(
        default_factory=list,
        description="Альтернативные формулировки (синонимы) вопроса",
    )
    answer: str = Field(..., description="Текст ответа")


class FaqListResponse(BaseModel):
    items: list[FaqItem] = Field(..., description="Список вопросов-ответов")


class FaqSettings(BaseModel):
    """Настройки FAQ-матчера."""

    similarity_threshold: float = Field(
        0.95,
        ge=0.5,
        le=1,
        description="Порог косинусного сходства для срабатывания FAQ (0..1)",
    )


class FaqSettingsUpdate(BaseModel):
    """Обновление настроек FAQ-матчера."""

    # Ниже 0.5 FAQ отвечал бы готовым текстом почти на любой вопрос.
    similarity_threshold: float = Field(ge=0.5, le=1)
