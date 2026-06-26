import csv
import io
import itertools

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.abbrev import AbbrevItem, AbbrevListResponse
from api.services.abbrev import AbbrevService
from db.postgres.db import get_async_session

SHORT_COLUMNS = ("Аббревиатура", "Short", "Сокращение")
FULL_COLUMNS = ("Расшифровка", "Full", "Полная форма")

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
async def create(item: AbbrevItem, service: AbbrevService = Depends(get_abbrev_service)):
    try:
        return await service.create(item)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _find_column(headers: list[str], candidates: tuple[str, ...], fallback: int) -> int:
    for candidate in candidates:
        if candidate in headers:
            return headers.index(candidate)
    return fallback


@router.post(
    "/upload",
    response_model=AbbrevListResponse,
    status_code=201,
    summary="Загрузить аббревиатуры из CSV",
)
async def upload_abbrev_csv(
    file: UploadFile = File(...),
    service: AbbrevService = Depends(get_abbrev_service),
):
    """
    Загружает аббревиатуры из CSV-файла (UPSERT по полю «Аббревиатура»).

    Ожидаются две колонки:
    - «Аббревиатура» / «Short» / «Сокращение»
    - «Расшифровка» / «Full» / «Полная форма»

    Поиск колонок ведётся по названию, иначе используются позиции 0 и 1.
    Если аббревиатура уже существует — её расшифровка обновляется.
    При повторе аббревиатуры внутри файла побеждает последнее значение.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Только CSV файлы поддерживаются (расширение .csv)"
        )

    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1251")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Не удалось определить кодировку файла. Используйте UTF-8.",
            )

    try:
        dialect = csv.Sniffer().sniff(text[:1024] if len(text) > 1024 else text)
        reader = csv.reader(io.StringIO(text), dialect=dialect)
    except Exception:
        reader = csv.reader(io.StringIO(text))

    first_row = next(reader, None)
    if not first_row or len(first_row) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Неверный формат CSV. Нужно хотя бы 2 колонки: "
                "'Аббревиатура', 'Расшифровка'."
            ),
        )

    # Первая строка — заголовок, только если содержит известное имя колонки.
    # Иначе это данные (файл без заголовка), и первую строку терять нельзя.
    known_columns = set(SHORT_COLUMNS) | set(FULL_COLUMNS)
    if any(cell.strip() in known_columns for cell in first_row):
        short_idx = _find_column(first_row, SHORT_COLUMNS, 0)
        full_idx = _find_column(first_row, FULL_COLUMNS, 1)
        data_rows = reader
    else:
        short_idx, full_idx = 0, 1
        data_rows = itertools.chain([first_row], reader)

    items_to_import: list[AbbrevItem] = []
    for row in data_rows:
        if not row:
            continue
        short = row[short_idx].strip() if len(row) > short_idx else ""
        full = row[full_idx].strip() if len(row) > full_idx else ""
        if not short or not full:
            continue
        items_to_import.append(AbbrevItem(short=short, full=full))

    if not items_to_import:
        raise HTTPException(
            status_code=400, detail="Не найдено валидных аббревиатур в файле."
        )

    try:
        await service.import_many(items_to_import)
        return AbbrevListResponse(items=await service.get_all())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения: {str(e)}")


@router.put(
    "/{item_id}", response_model=AbbrevItem, summary="Обновить аббревиатуру по ID"
)  # noqa: E501
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
async def delete(item_id: str, service: AbbrevService = Depends(get_abbrev_service)):
    try:
        await service.delete(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Abbreviation {item_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
