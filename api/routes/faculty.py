from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.faculty import (
    FacultyCreate,
    FacultyItem,
    FacultyUpdate,
    ProgramCreate,
    ProgramItem,
    ProgramUpdate,
)
from db.postgres.db import get_async_session
from db.postgres.models import Faculty, Program
from db.postgres.services.faculty import FacultyService

router = APIRouter(prefix="/faculties", tags=["faculties"])


def get_faculty_service(
    session: AsyncSession = Depends(get_async_session),
) -> FacultyService:
    return FacultyService(session)


def _program_item(program: Program) -> ProgramItem:
    return ProgramItem.model_validate(program)


def _faculty_item(faculty: Faculty, programs: Sequence[Program]) -> FacultyItem:
    return FacultyItem(
        id=faculty.id,
        name=faculty.name,
        aliases=list(faculty.aliases or []),
        is_active=faculty.is_active,
        programs=[_program_item(p) for p in programs],
    )


@router.get(
    "",
    response_model=list[FacultyItem],
    summary="Список факультетов с направлениями",
)
async def get_faculties(
    service: FacultyService = Depends(get_faculty_service),
) -> list[FacultyItem]:
    """Возвращает все факультеты (включая выключенные) с их направлениями."""
    directory = await service.load_directory(
        faculties_only_active=False, programs_only_active=False
    )
    return [
        _faculty_item(faculty, directory.programs_by_faculty.get(faculty.id, []))
        for faculty in directory.faculties
    ]


@router.post(
    "",
    response_model=FacultyItem,
    status_code=status.HTTP_201_CREATED,
    summary="Создать факультет",
)
async def create_faculty(
    data: FacultyCreate,
    service: FacultyService = Depends(get_faculty_service),
) -> FacultyItem:
    faculty = await service.create_faculty(
        name=data.name,
        aliases=data.aliases,
        is_active=data.is_active,
    )
    programs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    return _faculty_item(faculty, programs)


@router.put(
    "/{faculty_id}",
    response_model=FacultyItem,
    summary="Обновить факультет",
)
async def update_faculty(
    faculty_id: str,
    data: FacultyUpdate,
    service: FacultyService = Depends(get_faculty_service),
) -> FacultyItem:
    faculty = await service.update_faculty(
        faculty_id,
        name=data.name,
        aliases=data.aliases,
        is_active=data.is_active,
    )
    if faculty is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Факультет с ID {faculty_id} не найден",
        )
    programs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    return _faculty_item(faculty, programs)


@router.delete(
    "/{faculty_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить факультет",
)
async def delete_faculty(
    faculty_id: str,
    service: FacultyService = Depends(get_faculty_service),
) -> None:
    deleted = await service.delete_faculty(faculty_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Факультет с ID {faculty_id} не найден",
        )


@router.post(
    "/{faculty_id}/programs",
    response_model=ProgramItem,
    status_code=status.HTTP_201_CREATED,
    summary="Добавить направление факультету",
)
async def create_program(
    faculty_id: str,
    data: ProgramCreate,
    service: FacultyService = Depends(get_faculty_service),
) -> ProgramItem:
    faculty = await service.get_faculty_by_id(faculty_id)
    if faculty is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Факультет с ID {faculty_id} не найден",
        )
    try:
        program = await service.create_program(
            faculty_id=faculty_id,
            name=data.name,
            level=data.level,
            code=data.code,
            is_active=data.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return _program_item(program)


@router.put(
    "/programs/{program_id}",
    response_model=ProgramItem,
    summary="Обновить направление",
)
async def update_program(
    program_id: str,
    data: ProgramUpdate,
    service: FacultyService = Depends(get_faculty_service),
) -> ProgramItem:
    try:
        program = await service.update_program(
            program_id,
            name=data.name,
            level=data.level,
            code=data.code,
            is_active=data.is_active,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if program is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Направление с ID {program_id} не найдено",
        )
    return _program_item(program)


@router.delete(
    "/programs/{program_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить направление",
)
async def delete_program(
    program_id: str,
    service: FacultyService = Depends(get_faculty_service),
) -> None:
    deleted = await service.delete_program(program_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Направление с ID {program_id} не найдено",
        )
