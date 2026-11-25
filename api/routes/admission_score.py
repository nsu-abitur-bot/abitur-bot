from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.admission_score import (
    ImportRequest,
    ImportResponse,
    PreviewRequest,
    PreviewResponse,
    PreviewSummary,
    ScoreItem,
    ScoreRowSchema,
    ScoreUpdate,
    UnmatchedSample,
)
from config.university import is_nsu
from db.postgres.db import get_async_session
from db.postgres.models import AdmissionBasis, StudyForm
from db.postgres.services.admission_score import (
    AdmissionScoreConflict,
    AdmissionScoreService,
    ScoreChanges,
    ScoreRow,
)
from parser.scores import DEFAULT_SCORES_URL, parse_scores

router = APIRouter(prefix="/admission-scores", tags=["admission-scores"])

_UNMATCHED_SAMPLE_LIMIT = 20


def get_admission_score_service(
    session: AsyncSession = Depends(get_async_session),
) -> AdmissionScoreService:
    return AdmissionScoreService(session)


def _to_row(item: ScoreRowSchema) -> ScoreRow:
    return ScoreRow(
        faculty_name=item.faculty_name,
        program_name=item.program_name,
        code=item.code,
        year=item.year,
        admission_basis=item.admission_basis,
        study_form=item.study_form,
        passing_score=item.passing_score,
        average_score=item.average_score,
        level=item.level,
        seat_count=item.seat_count,
        source_url=item.source_url,
    )


@router.post(
    "/preview",
    response_model=PreviewResponse,
    summary="Распарсить страницу итогов приёма и показать строки",
)
async def preview_scores(
    data: PreviewRequest,
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> PreviewResponse:
    """Парсит страницу (без записи в БД) и считает сопоставимость со справочником."""
    if not is_nsu():
        raise HTTPException(409, "Парсер баллов НГУ недоступен для другого вуза")
    url = data.url or DEFAULT_SCORES_URL
    parsed = await parse_scores(url)

    rows: list[ScoreRowSchema] = []
    matched = 0
    unmatched_samples: list[UnmatchedSample] = []
    seen_unmatched: set[tuple[str, str]] = set()

    for row in parsed:
        rows.append(ScoreRowSchema.model_validate(row, from_attributes=True))
        program_id = await service.resolve_program_id(
            row.faculty_name, row.program_name, row.level
        )
        if program_id is not None:
            matched += 1
        else:
            key = (row.faculty_name, row.program_name)
            if key not in seen_unmatched:
                seen_unmatched.add(key)
                if len(unmatched_samples) < _UNMATCHED_SAMPLE_LIMIT:
                    unmatched_samples.append(
                        UnmatchedSample(
                            faculty_name=row.faculty_name,
                            program_name=row.program_name,
                        )
                    )

    summary = PreviewSummary(
        total=len(parsed),
        matched=matched,
        unmatched=len(parsed) - matched,
        unmatched_samples=unmatched_samples,
    )
    return PreviewResponse(rows=rows, summary=summary)


@router.post(
    "/import",
    response_model=ImportResponse,
    summary="Импортировать отревьюенные строки проходных баллов",
)
async def import_scores(
    data: ImportRequest,
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> ImportResponse:
    """Идемпотентный upsert строк, присланных админом после предпросмотра."""
    stats = await service.upsert_from_rows(_to_row(item) for item in data.rows)
    return ImportResponse(**stats)


@router.get(
    "",
    response_model=list[ScoreItem],
    summary="Выборка проходных баллов",
)
async def get_scores(
    faculty: Optional[str] = Query(None, description="Фильтр по факультету"),
    program: Optional[str] = Query(None, description="Фильтр по направлению"),
    year: Optional[int] = Query(None, description="Фильтр по году"),
    form: Optional[str] = Query(
        None, description="Совместимый фильтр по основанию приёма", deprecated=True
    ),
    admission_basis: AdmissionBasis | None = Query(
        None, description="Основание приёма: budget / paid"
    ),
    study_form: StudyForm | None = Query(None, description="Форма обучения"),
    only_active: bool = Query(
        False, description="Только активные факультеты и программы"
    ),
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> list[ScoreItem]:
    legacy_basis = form.strip().lower() if form is not None else None
    if admission_basis is not None and legacy_basis not in (None, admission_basis):
        raise HTTPException(422, "form и admission_basis должны совпадать")
    rows = await service.query_scores(
        faculty=faculty,
        program=program,
        year=year,
        admission_basis=admission_basis or legacy_basis,
        study_form=study_form,
        only_active=only_active,
    )
    return [ScoreItem(**row) for row in rows]


@router.get(
    "/years",
    response_model=list[int],
    summary="Список доступных годов",
)
async def get_years(
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> list[int]:
    return await service.get_available_years()


@router.patch(
    "/{score_id}",
    response_model=ScoreItem,
    summary="Уточнить классификацию, количество мест или источник",
)
async def update_score(
    score_id: str,
    data: ScoreUpdate,
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> ScoreItem:
    changes: ScoreChanges = {}
    if data.admission_basis is not None:
        changes["admission_basis"] = data.admission_basis
    if "study_form" in data.model_fields_set:
        changes["study_form"] = data.study_form
    if "seat_count" in data.model_fields_set:
        changes["seat_count"] = data.seat_count
    if "source_url" in data.model_fields_set:
        changes["source_url"] = data.source_url
    try:
        row = await service.update_score(score_id, changes)
    except AdmissionScoreConflict:
        raise HTTPException(
            409, "Для этой формы обучения и основания приёма уже есть запись"
        ) from None
    if row is None:
        raise HTTPException(404, "Запись о баллах не найдена")
    return ScoreItem(**row)
