from pydantic import BaseModel, Field


class FaqItem(BaseModel):
    """Схема для отдельного элемента FAQ."""

    question: str = Field(..., description="Основной вопрос")
    aliases: list[str] = Field(default_factory=list, description="Альтернативные формулировки (синонимы) вопроса")
    answer: str = Field(..., description="Текст ответа")


class FaqListResponse(BaseModel):
    """Схема ответа для списка FAQ-вопросов."""

    items: list[FaqItem] = Field(..., description="Список вопросов-ответов")
