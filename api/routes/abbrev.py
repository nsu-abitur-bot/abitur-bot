from fastapi import APIRouter, Depends, HTTPException

from api.schemas.abbrev import AbbrevItem, AbbrevListResponse
from api.services.abbrev import AbbrevService

router = APIRouter(prefix="/abbrev", tags=["Abbreviations"])


def get_abbrev_service() -> AbbrevService:
    return AbbrevService()


@router.get("", response_model=AbbrevListResponse, summary="Получить все аббревиатуры")
def get_all(service: AbbrevService = Depends(get_abbrev_service)):
    """Возвращает словарь всех аббревиатур."""
    return AbbrevListResponse(items=service.get_all())


@router.post(
    "", response_model=AbbrevItem, status_code=201, summary="Добавить аббревиатуру"
)
def create(item: AbbrevItem, service: AbbrevService = Depends(get_abbrev_service)):
    """Добавляет новую аббревиатуру в словарь."""
    try:
        return service.create(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{index}", response_model=AbbrevItem, summary="Обновить аббревиатуру по индексу"
)
def update(
    index: int, item: AbbrevItem, service: AbbrevService = Depends(get_abbrev_service)
):
    """Обновляет существующую аббревиатуру по индексу в списке."""
    try:
        return service.update(index, item)
    except IndexError:
        raise HTTPException(
            status_code=404, detail=f"Abbreviation at index {index} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{index}", status_code=204, summary="Удалить аббревиатуру по индексу")
def delete(index: int, service: AbbrevService = Depends(get_abbrev_service)):
    """Удаляет аббревиатуру по индексу."""
    try:
        service.delete(index)
    except IndexError:
        raise HTTPException(
            status_code=404, detail=f"Abbreviation at index {index} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
