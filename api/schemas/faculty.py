from typing import Optional

from pydantic import BaseModel, Field


class ProgramItem(BaseModel):
    """Направление подготовки (образовательная программа)."""

    id: str
    faculty_id: str
    name: str
    code: Optional[str] = None
    level: str
    is_active: bool

    class Config:
        from_attributes = True


class ProgramCreate(BaseModel):
    """Данные для создания направления подготовки."""

    name: str = Field(..., min_length=1, description="Название направления")
    level: str = Field(..., description="Уровень: bachelor / specialist / master")
    code: Optional[str] = Field(None, description="Код направления (ФГОС)")
    is_active: bool = Field(True, description="Активно ли направление")


class ProgramUpdate(BaseModel):
    """Данные для обновления направления подготовки."""

    name: Optional[str] = Field(None, min_length=1, description="Название направления")
    level: Optional[str] = Field(None, description="Уровень образования")
    code: Optional[str] = Field(None, description="Код направления (ФГОС)")
    is_active: Optional[bool] = Field(None, description="Активно ли направление")


class FacultyItem(BaseModel):
    """Факультет с его направлениями подготовки."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    is_active: bool
    programs: list[ProgramItem] = Field(default_factory=list)

    class Config:
        from_attributes = True


class FacultyCreate(BaseModel):
    """Данные для создания факультета."""

    name: str = Field(..., min_length=1, description="Название факультета")
    aliases: list[str] = Field(
        default_factory=list, description="Аббревиатуры и альтернативные названия"
    )
    is_active: bool = Field(True, description="Активен ли факультет")


class FacultyUpdate(BaseModel):
    """Данные для обновления факультета."""

    name: Optional[str] = Field(None, min_length=1, description="Название факультета")
    aliases: Optional[list[str]] = Field(
        None, description="Аббревиатуры и альтернативные названия"
    )
    is_active: Optional[bool] = Field(None, description="Активен ли факультет")
