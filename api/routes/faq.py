import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.schemas.faq import FaqItem, FaqListResponse
from api.services.faq import FaqService

router = APIRouter(prefix="/faq", tags=["FAQ Management"])


# Зависимость для получения сервиса (в будущем: кэширование / DI container)
def get_faq_service() -> FaqService:
    return FaqService()


@router.get("", response_model=FaqListResponse, summary="Получить все FAQ")
def get_all_faqs(service: FaqService = Depends(get_faq_service)):
    """Возвращает список всех вопросов и ответов FAQ, которые использует бот."""
    items = service.get_all()
    return FaqListResponse(items=items)


@router.post("", response_model=FaqItem, status_code=201, summary="Создать FAQ")
def create_faq(item: FaqItem, service: FaqService = Depends(get_faq_service)):
    """Создает новый вопрос FAQ и автоматически применяет для новых запросов к боту."""
    try:
        created = service.create(item)
        return created
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
        text = content.decode("utf-8-sig")  # Поддержка UTF-8 с BOM или без
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

    # Пытаемся найти колонки по названиям, иначе берем первые две
    q_idx = headers.index("Вопросы") if "Вопросы" in headers else 0
    a_idx = headers.index("Ответы") if "Ответы" in headers else 1

    items_to_create = []
    current_questions = []
    current_answer = ""

    for row in reader:
        # Если строка физически пустая (например, [])
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

        # Если в строке обе ячейки пусты, это тоже разделитель блоков
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
            if q:
                if q not in current_questions:
                    current_questions.append(q)
            if a and not current_answer:
                current_answer = a

    # Обрабатываем последний блок, если файл не заканчивался пустой строкой
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
        created = service.create_many(items_to_create)
        return FaqListResponse(items=created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения: {str(e)}")


@router.put("/{index}", response_model=FaqItem, summary="Обновить FAQ по индексу")
def update_faq(
    index: int, item: FaqItem, service: FaqService = Depends(get_faq_service)
):
    """Обновляет существующий FAQ элемент по его позиции (индексу) в списке."""
    try:
        updated = service.update(index, item)
        return updated
    except IndexError:
        raise HTTPException(
            status_code=404, detail=f"FAQ item at index {index} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{index}", status_code=204, summary="Удалить FAQ по индексу")
def delete_faq(index: int, service: FaqService = Depends(get_faq_service)):
    """Удаляет существующий FAQ элемент по его позиции (индексу) в списке."""
    try:
        service.delete(index)
    except IndexError:
        raise HTTPException(
            status_code=404, detail=f"FAQ item at index {index} not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
