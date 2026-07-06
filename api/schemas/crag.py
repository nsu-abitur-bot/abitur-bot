from pydantic import BaseModel, Field


class CragSettings(BaseModel):
    """Настройки корректирующего RAG (CRAG)."""

    enabled: bool = Field(
        True,
        description="Включён ли CRAG-слой (корректирующий RAG)",
    )
    relevance_threshold: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Порог релевантности чанка при LLM-грейдинге (0..1)",
    )
    min_chunks: int = Field(
        2,
        ge=0,
        le=100,
        description="Минимум чанков после фильтра, ниже которого пробуем доретрив",
    )
    allow_refine: bool = Field(
        True,
        description="Разрешить одну переформулировку запроса и доретрив",
    )
    use_faculty_table: bool = Field(
        True,
        description="Авторитетная фильтрация по таблице факультетов",
    )
    max_graded_chunks: int = Field(
        12,
        ge=1,
        le=100,
        description="Максимум чанков, отправляемых на LLM-грейдинг (латентность)",
    )


class CragSettingsUpdate(BaseModel):
    """Обновление настроек CRAG."""

    enabled: bool
    relevance_threshold: float = Field(ge=0, le=1)
    min_chunks: int = Field(ge=0, le=100)
    allow_refine: bool
    use_faculty_table: bool
    max_graded_chunks: int = Field(ge=1, le=100)
