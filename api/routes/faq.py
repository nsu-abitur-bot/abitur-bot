import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.faq import FaqItem, FaqListResponse
from api.services.faq import FaqService
from db.postgres.db import get_async_session

router = APIRouter(prefix="/faq", tags=["FAQ Management"])


def get_faq_service(session: AsyncSession = Depends(get_async_session)) -> FaqService:
    return FaqService(session)


@router.get("", response_model=FaqListResponse, summary="Получить все FAQ")
async def get_all_faqs(service: FaqService = Depends(get_faq_service)):
    items = await service.get_all()
    return FaqListResponse(items=items)


@router.post("", response_model=FaqItem, status_code=201, summary="Создать FAQ")
async def create_faq(item: FaqItem, service: FaqService = Depends(get_faq_service)):
    try:
        return await service.create(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upload",
    response_model=FaqListResponse,
    status_code=201,
    summary="Загрузить FAQ из CSV",
)
async def upload_faq_csv(
    file: UploadFile = File(...), service: FaqService = Depends(get_faq_service)
):
    """
    Загружает вопросы и ответы FAQ из CSV-файла.
    Ожидается CSV файл с колонками "Вопросы" и "Ответы".
    Пустая строка означает, что начинается новый вопрос.
    Первый вопрос в блоке становится основным,
    остальные - альтернативными формулировками (aliases).
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

    headers = next(reader, None)
    if not headers or len(headers) < 2:
        raise HTTPException(
            status_code=400,
            detail="Неверный формат CSV. Нужно хотя бы 2 колонки: 'Вопросы', 'Ответы'.",
        )

    q_idx = headers.index("Вопросы") if "Вопросы" in headers else 0
    a_idx = headers.index("Ответы") if "Ответы" in headers else 1

    items_to_create = []
    current_questions: list[str] = []
    current_answer = ""

    for row in reader:
        if not row:
            if current_questions and current_answer:
                items_to_create.append(
                    FaqItem(
                        question=current_questions[0],
                        aliases=current_questions[1:],
                        answer=current_answer,
                    )
                )
            current_questions = []
            current_answer = ""
            continue

        q = row[q_idx].strip() if len(row) > q_idx else ""
        a = row[a_idx].strip() if len(row) > a_idx else ""

        if not q and not a:
            if current_questions and current_answer:
                items_to_create.append(
                    FaqItem(
                        question=current_questions[0],
                        aliases=current_questions[1:],
                        answer=current_answer,
                    )
                )
            current_questions = []
            current_answer = ""
        else:
            if q and q not in current_questions:
                current_questions.append(q)
            if a and not current_answer:
                current_answer = a

    if current_questions and current_answer:
        items_to_create.append(
            FaqItem(
                question=current_questions[0],
                aliases=current_questions[1:],
                answer=current_answer,
            )
        )

    if not items_to_create:
        raise HTTPException(
            status_code=400, detail="Не найдено валидных вопросов/ответов в файле."
        )

    try:
        created = await service.create_many(items_to_create)
        return FaqListResponse(items=created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения: {str(e)}")


@router.put("/{item_id}", response_model=FaqItem, summary="Обновить FAQ по ID")
async def update_faq(
    item_id: str, item: FaqItem, service: FaqService = Depends(get_faq_service)
):
    try:
        return await service.update(item_id, item)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"FAQ item {item_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{item_id}", status_code=204, summary="Удалить FAQ по ID")
async def delete_faq(item_id: str, service: FaqService = Depends(get_faq_service)):
    try:
        await service.delete(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"FAQ item {item_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
