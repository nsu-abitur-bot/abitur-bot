from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.abbrev import AbbrevItem, AbbrevListResponse
from api.services.abbrev import AbbrevService
from db.postgres.db import get_async_session

router = APIRouter(prefix="/abbrev", tags=["Abbreviations"])


def get_abbrev_service(
    session: AsyncSession = Depends(get_async_session),
) -> AbbrevService:
    return AbbrevService(session)


@router.get("", response_model=AbbrevListResponse, summary="Получить все аббревиатуры")
async def get_all(service: AbbrevService = Depends(get_abbrev_service)):
    return AbbrevListResponse(items=await service.get_all())


@router.post(
    "", response_model=AbbrevItem, status_code=201, summary="Добавить аббревиатуру"
)
async def create(
    item: AbbrevItem, service: AbbrevService = Depends(get_abbrev_service)
):
    try:
        return await service.create(item)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{item_id}", response_model=AbbrevItem, summary="Обновить аббревиатуру по ID")  # noqa: E501
async def update(
    item_id: str,
    item: AbbrevItem,
    service: AbbrevService = Depends(get_abbrev_service),
):
    try:
        return await service.update(item_id, item)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Abbreviation {item_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{item_id}", status_code=204, summary="Удалить аббревиатуру по ID")
async def delete(
    item_id: str, service: AbbrevService = Depends(get_abbrev_service)
):
    try:
        await service.delete(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Abbreviation {item_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
