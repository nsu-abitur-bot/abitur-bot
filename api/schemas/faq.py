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
