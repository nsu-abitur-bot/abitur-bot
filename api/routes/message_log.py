from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.message_log import (
    AnswerStatsResponse,
    FaqHitStatsResponse,
    MessageLogListResponse,
    MessageLogQueryParams,
    MessageLogResponse,
    RequestCountStatsResponse,
    TokenUsageStatsResponse,
)
from api.schemas.popular_questions import (
    PopularQuestion,
    PopularQuestionsResponse,
    PopularQuestionsSnapshotResponse,
)
from api.services.popular_statistics import PopularStatistics, get_popular_statistics
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
    "/faq-stats",
    response_model=FaqHitStatsResponse,
    summary="Как часто FAQ-слой отвечает без обращения к модели",
)
async def get_faq_hit_stats(
    start: datetime | None = Query(None, description="Начало периода (ISO 8601)"),
    end: datetime | None = Query(None, description="Конец периода (ISO 8601)"),
    group_by: str = Query(
        "day",
        description="Группировка: hour, day, week, month",
    ),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """Доля вопросов, закрытых готовыми ответами FAQ (трек качества).
    Порог матчера настраивается через PUT /settings/faq."""
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
        stats = await log_service.get_faq_hit_stats(
            start=start,
            end=end,
            group_by=group_by,
        )
        return FaqHitStatsResponse(
            total_questions=stats["total_questions"],
            total_hits=stats["total_hits"],
            hit_rate=stats["hit_rate"],
            group_by=group_by,
            start=start,
            end=end,
            buckets=stats["buckets"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики FAQ: {str(e)}",
        )


@router.get(
    "/answer-stats",
    response_model=AnswerStatsResponse,
    summary="Метрики ответов: latency, доля неотвеченных, ошибки",
)
async def get_answer_stats(
    start: datetime | None = Query(None, description="Начало периода (ISO 8601)"),
    end: datetime | None = Query(None, description="Конец периода (ISO 8601)"),
    group_by: str = Query(
        "day",
        description="Группировка: hour, day, week, month",
    ),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """Наблюдаемость ответов по периодам (#242): время отклика (p50/p95),
    доля неотвеченных и ошибок. Наполняется с момента деплоя: метрики
    пишутся пайплайном в message_metadata каждой записи llm_response."""
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
        stats = await log_service.get_answer_stats(
            start=start,
            end=end,
            group_by=group_by,
        )
        return AnswerStatsResponse(
            total_questions=stats["total_questions"],
            total_answers=stats["total_answers"],
            total_no_answer=stats["total_no_answer"],
            no_answer_share=stats["no_answer_share"],
            total_errors=stats["total_errors"],
            error_share=stats["error_share"],
            group_by=group_by,
            start=start,
            end=end,
            buckets=stats["buckets"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения метрик ответов: {str(e)}",
        )


@router.get(
    "/token-stats",
    response_model=TokenUsageStatsResponse,
    summary="Статистика потребления токенов",
)
async def get_token_stats(
    start: datetime | None = Query(None, description="Начало периода (ISO 8601)"),
    end: datetime | None = Query(None, description="Конец периода (ISO 8601)"),
    group_by: str = Query(
        "day",
        description="Группировка: hour, day, week, month",
    ),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """Возвращает суммарное потребление токенов LLM, сгруппированное по периодам."""
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
        stats = await log_service.get_token_usage_stats(
            start=start,
            end=end,
            group_by=group_by,
        )
        return TokenUsageStatsResponse(
            total=stats["total"],
            group_by=group_by,
            start=start,
            end=end,
            buckets=stats["buckets"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики токенов: {str(e)}",
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
    "/popular",
    response_model=PopularQuestionsSnapshotResponse,
    summary="Получить самые популярные вопросы",
)
async def get_popular_questions(
    limit: int = Query(10, ge=1, le=100),
    raw_limit: int = Query(500, ge=1, le=5000),
    similarity_threshold: float = Query(0.86, ge=0.5, le=1.0),
    statistics: PopularStatistics = Depends(get_popular_statistics),
):
    """
    Получает самые часто задаваемые вопросы пользователей.

    Читает подготовленный фоновым обработчиком снимок. `pending` означает,
    что снимок ещё не подготовлен; GET не вызывает модели и кластеризацию.
    Новые комбинации параметров подхватывает следующий фоновый обход.

    - **limit**: максимальное количество возвращаемых кластеров вопросов
    - **raw_limit**: сколько исходных формулировок взять для кластеризации
    - **similarity_threshold**: порог семантической близости формулировок
    """
    try:
        snapshot = await statistics.read(
            limit=limit,
            raw_limit=raw_limit,
            similarity_threshold=similarity_threshold,
        )
        return PopularQuestionsSnapshotResponse.model_validate(snapshot)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения популярных вопросов: {str(e)}"
        )


@router.get(
    "/unanswered",
    response_model=PopularQuestionsResponse,
    summary="Очередь вопросов без ответа",
)
async def get_unanswered_questions(
    limit: int = Query(50, ge=1, le=200),
    raw_limit: int = Query(500, ge=1, le=5000),
    log_service: MessageLogService = Depends(get_message_log_service),
):
    """Вопросы, на которые бот честно ответил «не нашёл информации» (#258).

    Пайплайн пишет их с типом `no_answer`; здесь они сгруппированы с частотой
    и вариантами формулировок — это очередь на наполнение базы знаний.
    Сервисные сбои («сервис недоступен») в очередь не попадают: это ошибки
    инфраструктуры, а не пробелы контента.
    """
    try:
        groups = await log_service.get_question_groups(
            raw_limit=raw_limit,
            message_type="no_answer",
        )
        return PopularQuestionsResponse(
            questions=[
                PopularQuestion(
                    question=item["question"],
                    count=item["count"],
                    variants=item["variants"],
                )
                for item in groups[:limit]
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения вопросов без ответа: {str(e)}"
        )
