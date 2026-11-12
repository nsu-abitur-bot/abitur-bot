"""Отчёт о пробелах наполнения для редактора (#262)."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.gap_report import build_gap_report
from db.postgres.db import get_async_session

router = APIRouter(prefix="/reports", tags=["Reports"])


class ReportTopic(BaseModel):
    topic: str
    count: int


class ReportQuestion(BaseModel):
    question: str
    count: int
    variants: List[str] = Field(default_factory=list)


class ReportQuality(BaseModel):
    questions: int
    answers: int
    no_answer: int
    no_answer_share: Optional[float] = None
    errors: int
    error_share: Optional[float] = None


class DislikedQuestion(BaseModel):
    question: str
    negative: int
    rated: int


class GapReportResponse(BaseModel):
    period_days: int
    generated_at: datetime
    topics: List[ReportTopic]
    popular: List[ReportQuestion]
    unanswered: List[ReportQuestion]
    quality: ReportQuality
    disliked: List[DislikedQuestion]


@router.get(
    "/gaps",
    response_model=GapReportResponse,
    summary="Отчёт о пробелах: темы, популярное, неотвеченное, качество",
)
async def get_gap_report(
    days: int = Query(30, ge=1, le=365, description="Период отчёта в днях"),
    limit: int = Query(10, ge=1, le=50, description="Размер каждого топа"),
    session: AsyncSession = Depends(get_async_session),
):
    """Собирается из логов и оценок без вызовов моделей (#262).

    Разделы: распределение тем, топ вопросов, очередь неотвеченных (#
    442), сводка качества ответов (#444) и вопросы с негативной оценкой.
    """
    try:
        report = await build_gap_report(session, days=days, limit=limit)
        return GapReportResponse.model_validate(report)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка построения отчёта о пробелах: {str(e)}"
        )
