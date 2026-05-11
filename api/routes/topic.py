from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.topic import TopicCreate, TopicListResponse, TopicResponse, TopicUpdate
from db.postgres.db import get_db
from db.postgres.services.topic import TopicService

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/", response_model=TopicListResponse, summary="Получить все темы")
async def get_topics(db: AsyncSession = Depends(get_db)):
    """Получить список всех активных тем."""
    topic_service = TopicService(db)
    topics = await topic_service.get_all_active_topics()
    return TopicListResponse(
        topics=[
            TopicResponse.model_validate(topic) for topic in topics
        ],
        total=len(topics),
    )


@router.post(
    "/",
    response_model=TopicResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новую тему",
)
async def create_topic(
    topic_data: TopicCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Создает новую тему.
    
    - **label**: уникальное название темы
    - **description**: опциональное описание
    - **is_active**: активна ли тема при создании (по умолчанию True)
    """
    topic_service = TopicService(db)
    
    # Проверяем, не существует ли уже тема с таким названием
    existing_topic = await topic_service.get_topic_by_label(topic_data.label)
    if existing_topic:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Тема с названием '{topic_data.label}' уже существует",
        )
    
    topic = await topic_service.create_topic(
        label=topic_data.label,
        description=topic_data.description,
        is_active=topic_data.is_active,
    )
    return TopicResponse.model_validate(topic)


@router.get(
    "/{topic_id}",
    response_model=TopicResponse,
    summary="Получить тему по ID",
)
async def get_topic(
    topic_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Получает тему по её ID."""
    topic_service = TopicService(db)
    topic = await topic_service.get_topic_by_id(topic_id)
    
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Тема с ID {topic_id} не найдена",
        )
    
    return TopicResponse.model_validate(topic)


@router.put(
    "/{topic_id}",
    response_model=TopicResponse,
    summary="Обновить тему",
)
async def update_topic(
    topic_id: int,
    topic_data: TopicUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Обновляет тему по ID.
    
    - **label**: новое название (опционально)
    - **description**: новое описание (опционально)
    - **is_active**: изменить статус активности (опционально)
    """
    topic_service = TopicService(db)
    
    # Проверяем, существует ли тема
    topic = await topic_service.get_topic_by_id(topic_id)
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Тема с ID {topic_id} не найдена",
        )
    
    # Проверяем, не существует ли уже тема с новым названием (если оно меняется)
    if topic_data.label and topic_data.label != topic.label:
        existing_topic = await topic_service.get_topic_by_label(topic_data.label)
        if existing_topic:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Тема с названием '{topic_data.label}' уже существует",
            )
    
    updated_topic = await topic_service.update_topic(
        topic_id=topic_id,
        label=topic_data.label,
        description=topic_data.description,
        is_active=topic_data.is_active,
    )
    
    return TopicResponse.model_validate(updated_topic)


@router.delete(
    "/{topic_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить тему",
)
async def delete_topic(
    topic_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Удаляет (деактивирует) тему по ID.
    
    Примечание: тема не удаляется из БД, а деактивируется (is_active = False)
    для сохранения целостности исторических данных.
    """
    topic_service = TopicService(db)
    
    # Проверяем, существует ли тема
    topic = await topic_service.get_topic_by_id(topic_id)
    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Тема с ID {topic_id} не найдена",
        )
    
    success = await topic_service.delete_topic(topic_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при удалении темы",
        )
