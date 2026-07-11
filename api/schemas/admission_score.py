from typing import Optional

from pydantic import BaseModel, Field


class ScoreRowSchema(BaseModel):
    """Строка проходного балла (сырая, до/после разрешения в program_id)."""

    faculty_name: str = Field(..., description="Название факультета/института")
    program_name: str = Field(..., description="Название направления подготовки")
    code: Optional[str] = Field(None, description="Код направления (ФГОС)")
    year: int = Field(..., description="Год приёма")
    form: str = Field(..., description="Форма обучения: budget / paid")
    passing_score: Optional[int] = Field(None, description="Проходной балл")
    average_score: Optional[float] = Field(None, description="Средний балл")
    level: str = Field("bachelor", description="Уровень: bachelor / specialist")


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


class ScoreItem(BaseModel):
    """Плоская запись проходного балла для выборки."""

    faculty_name: str
    program_name: str
    code: Optional[str] = None
    year: int
    form: str
    passing_score: Optional[int] = None
    average_score: Optional[float] = None
