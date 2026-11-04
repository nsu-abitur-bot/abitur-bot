from datetime import UTC, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class FaqItem(BaseModel):
    id: Optional[str] = Field(None, description="UUID записи (заполняется при ответе)")
    question: str = Field(..., description="Основной вопрос")
    aliases: list[str] = Field(
        default_factory=list,
        description="Альтернативные формулировки (синонимы) вопроса",
    )
    answer: str = Field(..., description="Текст ответа")
    verification_status: Literal["draft", "confirmed"] = Field(
        "draft", description="Черновик или подтверждено человеком"
    )
    source_document: str | None = Field(None, description="Документ-источник")
    source_location: str | None = Field(None, description="Место в документе")
    confirmed_by: str | None = Field(None, description="Имя подтвердившего")
    confirmed_at: datetime | None = Field(
        None,
        description="Дата подтверждения в UTC; дата без часового пояса считается UTC",
    )

    @field_validator("source_document", "source_location", "confirmed_by")
    @classmethod
    def normalize_provenance(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("confirmed_at")
    @classmethod
    def normalize_confirmation_date(cls, value: datetime | None) -> datetime | None:
        # PostgreSQL stores this field without a timezone; naive UI dates mean UTC.
        if value is not None and value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    @model_validator(mode="after")
    def validate_confirmation(self) -> "FaqItem":
        if self.verification_status == "confirmed" and (
            self.confirmed_at is None
            or not (self.confirmed_by or (self.source_document and self.source_location))
        ):
            raise ValueError(
                "Для подтверждения укажите дату и имя проверившего "
                "либо документ и место в нём"
            )
        return self


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
