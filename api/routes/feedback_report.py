from fastapi import APIRouter, Depends, HTTPException

from api.schemas.feedback_report import (
    FeedbackReportListResponse,
    FeedbackReportQueryParams,
    FeedbackReportResponse,
    FeedbackReportStatusUpdate,
)
from db.postgres.db import get_async_session
from db.postgres.services.feedback_report import FeedbackReportService

router = APIRouter(prefix="/feedback", tags=["Feedback"])


def get_feedback_report_service(
    session=Depends(get_async_session),
) -> FeedbackReportService:
    return FeedbackReportService(session)


@router.get(
    "",
    response_model=FeedbackReportListResponse,
    summary="Получить обращения обратной связи",
)
async def list_feedback_reports(
    params: FeedbackReportQueryParams = Depends(),
    service: FeedbackReportService = Depends(get_feedback_report_service),
) -> FeedbackReportListResponse:
    reports, total = await service.list_reports(
        status=params.status,
        user_id=params.user_id,
        session_id=params.session_id,
        limit=params.limit,
        offset=params.offset,
    )
    return FeedbackReportListResponse(
        reports=reports,
        total=total,
        limit=params.limit,
        offset=params.offset,
    )


@router.get(
    "/{report_id}",
    response_model=FeedbackReportResponse,
    summary="Получить обращение обратной связи",
)
async def get_feedback_report(
    report_id: int,
    service: FeedbackReportService = Depends(get_feedback_report_service),
) -> FeedbackReportResponse:
    report = await service.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Feedback report not found")
    return FeedbackReportResponse.model_validate(report)


@router.patch(
    "/{report_id}",
    response_model=FeedbackReportResponse,
    summary="Обновить статус обращения",
)
async def update_feedback_report_status(
    report_id: int,
    payload: FeedbackReportStatusUpdate,
    service: FeedbackReportService = Depends(get_feedback_report_service),
) -> FeedbackReportResponse:
    report = await service.update_status(report_id, payload.status)
    if report is None:
        raise HTTPException(status_code=404, detail="Feedback report not found")
    return FeedbackReportResponse.model_validate(report)
