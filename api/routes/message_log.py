from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.message_log import (
    MessageLogListResponse,
    MessageLogQueryParams,
    MessageLogResponse,
    RequestCountStatsResponse,
)
from api.schemas.popular_questions import PopularQuestion, PopularQuestionsResponse
from db.postgres.db import get_async_session
from db.postgres.services.message_log import MessageLogService

router = APIRouter(prefix="/logs", tags=["Message Logs"])


def get_message_log_service(session=Depends(get_async_session)) -> MessageLogService:
    """Получает сервис для работы с логами сообщений."""
    return MessageLogService(session)


@router.get(
    "/request-stats",
    response_model=RequestCountStatsResponse,
    summary="Статистика количества запросов",
)
async def get_request_stats(
    start: datetime | None = Query(None, description="Начало периода (ISO 8601)"),
    end: datetime | None = Query(None, description="Конец периода (ISO 8601)"),
    group_by: str = Query(
        "day",
        description="Группировка: hour, day, week, month",
    ),
    message_type: str = Query("user_input", description="Тип сообщения для статистики"),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    allowed_groups = {"hour", "day", "week", "month"}
    if group_by not in allowed_groups:
        raise HTTPException(
            status_code=400,
            detail=f"group_by должен быть одним из {sorted(allowed_groups)}",
        )

    if start is not None and end is not None and start > end:
        raise HTTPException(
            status_code=400,
            detail="start не может быть позже end",
        )

    try:
        stats = await log_service.get_request_count_stats(
            start=start,
            end=end,
            group_by=group_by,
            message_type=message_type,
        )
        return RequestCountStatsResponse(
            total=stats["total"],
            group_by=group_by,
            start=start,
            end=end,
            buckets=stats["buckets"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}",
        )


@router.get(
    "/",
    response_model=MessageLogListResponse,
    summary="Получить логи сообщений",
)
async def get_message_logs(
    params: MessageLogQueryParams = Depends(),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """
    Получает логи сообщений с фильтрацией.

    - **user_id**: фильтр по ID пользователя
    - **session_id**: фильтр по ID сессии
    - **message_type**: фильтр по типу сообщения
    - **limit**: количество записей (макс. 1000)
    - **offset**: сдвиг для пагинации
    """
    try:
        if params.user_id:
            logs = await log_service.get_logs_by_user(
                user_id=params.user_id,
                limit=params.limit,
                offset=params.offset,
            )
        elif params.session_id:
            logs = await log_service.get_logs_by_session(
                session_id=params.session_id,
                limit=params.limit,
                offset=params.offset,
            )
        elif params.message_type:
            logs = await log_service.get_logs_by_type(
                message_type=params.message_type,
                limit=params.limit,
                offset=params.offset,
            )
        else:
            logs = await log_service.get_recent_logs(
                limit=params.limit,
                offset=params.offset,
            )

        return MessageLogListResponse(
            logs=[MessageLogResponse.model_validate(log) for log in logs],
            total=len(logs),
            limit=params.limit,
            offset=params.offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения логов: {str(e)}")


@router.get(
    "/session/{session_id}",
    response_model=MessageLogListResponse,
    summary="Получить логи по сессии",
)
async def get_session_logs(
    session_id: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """Получает все логи для конкретной сессии."""
    try:
        logs = await log_service.get_logs_by_session(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

        return MessageLogListResponse(
            logs=[MessageLogResponse.model_validate(log) for log in logs],
            total=len(logs),
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения логов сессии: {str(e)}"
        )


@router.get(
    "/user/{user_id}",
    response_model=MessageLogListResponse,
    summary="Получить логи по пользователю",
)
async def get_user_logs(
    user_id: int,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    log_service: MessageLogService = Depends(get_message_log_service),
) -> MessageLogListResponse:
    """Получает все логи для конкретного пользователя."""
    try:
        logs = await log_service.get_logs_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return MessageLogListResponse(
            logs=[MessageLogResponse.model_validate(log) for log in logs],
            total=len(logs),
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения логов пользователя: {str(e)}"
        )


@router.get(
    "/type/{message_type}",
    response_model=MessageLogListResponse,
    summary="Получить логи по типу сообщения",
)
async def get_type_logs(
    message_type: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """
    Получает логи по типу сообщения.

    Возможные типы:
    - `user_input` - входящие сообщения от пользователей
    - `rag_context` - контекст полученный из RAG
    - `llm_response` - ответы от LLM
    - `faq_match` - совпадения из FAQ
    """
    try:
        logs = await log_service.get_logs_by_type(
            message_type=message_type,
            limit=limit,
            offset=offset,
        )

        return MessageLogListResponse(
            logs=[MessageLogResponse.model_validate(log) for log in logs],
            total=len(logs),
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения логов по типу: {str(e)}"
        )


@router.get(
    "/popular",
    response_model=PopularQuestionsResponse,
    summary="Получить самые популярные вопросы",
)
async def get_popular_questions(
    limit: int = Query(10, ge=1, le=100),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """
    Получает самые часто задаваемые вопросы пользователей.

    - **limit**: максимальное количество возвращаемых вопросов
    """
    try:
        popular = await log_service.get_popular_questions(limit=limit)
        return PopularQuestionsResponse(
            questions=[
                PopularQuestion(question=item["question"], count=item["count"])
                for item in popular
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения популярных вопросов: {str(e)}"
        )
