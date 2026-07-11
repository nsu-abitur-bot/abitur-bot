from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.admission_score import (
    ImportRequest,
    ImportResponse,
    PreviewRequest,
    PreviewResponse,
    PreviewSummary,
    ScoreItem,
    ScoreRowSchema,
    UnmatchedSample,
)
from db.postgres.db import get_async_session
from db.postgres.services.admission_score import AdmissionScoreService, ScoreRow
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
        form=item.form,
        passing_score=item.passing_score,
        average_score=item.average_score,
        level=item.level,
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
    url = data.url or DEFAULT_SCORES_URL
    parsed = await parse_scores(url)

    rows: list[ScoreRowSchema] = []
    matched = 0
    unmatched_samples: list[UnmatchedSample] = []
    seen_unmatched: set[tuple[str, str]] = set()

    for row in parsed:
        rows.append(
            ScoreRowSchema(
                faculty_name=row.faculty_name,
                program_name=row.program_name,
                code=row.code,
                year=row.year,
                form=row.form,
                passing_score=row.passing_score,
                average_score=row.average_score,
                level=row.level,
            )
        )
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
    form: Optional[str] = Query(None, description="Форма: budget / paid"),
    service: AdmissionScoreService = Depends(get_admission_score_service),
) -> list[ScoreItem]:
    rows = await service.query_scores(
        faculty=faculty, program=program, year=year, form=form
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
