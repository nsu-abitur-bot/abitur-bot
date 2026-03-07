from fastapi import APIRouter, Depends, HTTPException

from api.schemas.faq import FaqItem, FaqListResponse
from api.services.faq import FaqService

router = APIRouter(
    prefix="/faq",
    tags=["FAQ Management"]
)

# Зависимость для получения сервиса (в будущем можно добавить кэширование / DI container)
def get_faq_service() -> FaqService:
    return FaqService()


@router.get("", response_model=FaqListResponse, summary="Получить все FAQ")
def get_all_faqs(service: FaqService = Depends(get_faq_service)):
    """Возвращает список всех вопросов и ответов FAQ, которые использует бот."""
    items = service.get_all()
    return FaqListResponse(items=items)


@router.post("", response_model=FaqItem, status_code=201, summary="Создать FAQ")
def create_faq(item: FaqItem, service: FaqService = Depends(get_faq_service)):
    """Создает новый вопрос FAQ и автоматически применяет его для новых запросов к боту."""
    try:
        created = service.create(item)
        return created
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{index}", response_model=FaqItem, summary="Обновить FAQ по индексу")
def update_faq(index: int, item: FaqItem, service: FaqService = Depends(get_faq_service)):
    """Обновляет существующий FAQ элемент по его позиции (индексу) в списке."""
    try:
        updated = service.update(index, item)
        return updated
    except IndexError:
        raise HTTPException(status_code=404, detail=f"FAQ item at index {index} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{index}", status_code=204, summary="Удалить FAQ по индексу")
def delete_faq(index: int, service: FaqService = Depends(get_faq_service)):
    """Удаляет существующий FAQ элемент по его позиции (индексу) в списке."""
    try:
        service.delete(index)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"FAQ item at index {index} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
