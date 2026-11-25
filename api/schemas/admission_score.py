from typing import Annotated, Any, Optional

from pydantic import (
    AliasChoices,
    AnyHttpUrl,
    BaseModel,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from db.postgres.models import AdmissionBasis, EducationLevel, StudyForm

SeatCount = Annotated[
    int,
    Field(
        strict=True,
        ge=0,
        le=2**31 - 1,
        description="Объявленное количество мест. null — неизвестно",
    ),
]

_SOURCE_URL = TypeAdapter(AnyHttpUrl)


class ScoreSource(BaseModel):
    source_url: str | None = Field(None, max_length=500, description="URL источника")

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str | None) -> str | None:
        if value is not None:
            _SOURCE_URL.validate_python(value)
        return value


class ScoreClassification(BaseModel):
    admission_basis: str = Field(
        validation_alias=AliasChoices("admission_basis", "form"),
        description="Основание приёма: budget / paid",
    )
    study_form: StudyForm | None = Field(
        None, description="Форма обучения. null — неизвестна"
    )

    @model_validator(mode="before")
    @classmethod
    def check_legacy_form(cls, data: Any) -> Any:
        if isinstance(data, dict):
            basis = data.get("admission_basis")
            legacy = data.get("form")
            if isinstance(basis, str) and isinstance(legacy, str):
                if basis.strip().lower() != legacy.strip().lower():
                    raise ValueError("form и admission_basis должны совпадать")
        return data

    @computed_field(json_schema_extra={"deprecated": True})
    @property
    def form(self) -> str:
        """Совместимое поле v1 для основания приёма."""
        return self.admission_basis


class ScoreRowSchema(ScoreClassification, ScoreSource):
    """Строка проходного балла (сырая, до/после разрешения в program_id)."""

    faculty_name: str = Field(..., description="Название факультета/института")
    program_name: str = Field(..., description="Название направления подготовки")
    code: Optional[str] = Field(None, description="Код направления (ФГОС)")
    year: int = Field(..., description="Год приёма")
    passing_score: Optional[float] = Field(None, description="Проходной балл")
    average_score: Optional[float] = Field(None, description="Средний балл")
    seat_count: SeatCount | None = None
    level: str = Field(
        "bachelor", description="Уровень: bachelor / specialist / master / spo (СПО)"
    )


class UnmatchedSample(BaseModel):
    """Пример строки, не сопоставленной со справочником факультетов."""

    faculty_name: str
    program_name: str


class PreviewSummary(BaseModel):
    """Сводка предпросмотра распарсенных строк."""

    total: int = Field(..., description="Всего распарсено строк")
    matched: int = Field(..., description="Строк, сопоставленных со справочником")
    unmatched: int = Field(..., description="Строк без сопоставления")
    unmatched_samples: list[UnmatchedSample] = Field(
        default_factory=list,
        description="Примеры несопоставленных (факультет, направление)",
    )


class PreviewRequest(BaseModel):
    """Запрос на предпросмотр парсинга страницы итогов приёма."""

    url: Optional[str] = Field(
        None, description="URL страницы (по умолчанию — страница НГУ)"
    )


class PreviewResponse(BaseModel):
    """Результат предпросмотра: распарсенные строки и сводка."""

    rows: list[ScoreRowSchema] = Field(default_factory=list)
    summary: PreviewSummary


class ImportRequest(BaseModel):
    """Запрос на импорт отревьюенных строк проходных баллов."""

    rows: list[ScoreRowSchema] = Field(default_factory=list)


class ImportResponse(BaseModel):
    """Статистика идемпотентного upsert."""

    created: int
    updated: int
    skipped: int


class ScoreItem(ScoreClassification):
    """Плоская запись проходного балла для выборки."""

    id: str
    faculty_name: str
    program_name: str
    level: EducationLevel = Field(..., description="Уровень образования программы")
    code: Optional[str] = None
    year: int
    passing_score: Optional[float] = None
    average_score: Optional[float] = None

    seat_count: SeatCount | None = None
    source_url: str | None = None


class ScoreUpdate(ScoreSource):
    admission_basis: AdmissionBasis | None = None
    study_form: StudyForm | None = None
    seat_count: SeatCount | None = None

    @field_validator("admission_basis")
    @classmethod
    def require_basis_if_provided(cls, value: AdmissionBasis | None) -> AdmissionBasis:
        if value is None:
            raise ValueError("Основание приёма обязательно")
        return value
