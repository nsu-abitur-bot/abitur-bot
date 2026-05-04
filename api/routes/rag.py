import csv
import io
import logging
from urllib.parse import unquote

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import HttpUrl

from abbrev.expander import get_abbrev_expander
from api.schemas.rag import (
    ConfirmUploadRequest,
    CsvImportPreviewResponse,
    CsvImportPreviewResult,
    CsvImportResponse,
    CsvImportResult,
    ParsedDocument,
    ParsedPageResult,
    RagDocumentContentResponse,
    RagDocumentListResponse,
    RagUploadResponse,
)
from api.services.rag_upload import RagUploadService
from api.services.url_to_rag import parse_and_save_url
from llm.preprocessor import clean_and_structure_text, generate_title_from_text
from parser.nsu import parse_page
from parser.url import process_pdf_bytes
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG Management"])


def get_rag_upload_service() -> RagUploadService:
    return RagUploadService()


@router.post(
    "/upload", response_model=RagUploadResponse, summary="Загрузить документы в RAG"
)
async def upload_documents_to_rag(
    files: list[UploadFile] = File(..., description="Файлы для индексации в RAG"),
    service: RagUploadService = Depends(get_rag_upload_service),
) -> RagUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        results = await service.ingest_files(files=files, graph_id=DEFAULT_GRAPH_ID)
    except Exception:
        raise HTTPException(status_code=503, detail="RAG ingestion unavailable")

    indexed_count = sum(1 for result in results if result.status == "indexed")
    skipped_count = len(results) - indexed_count

    return RagUploadResponse(
        accepted_formats=service.accepted_formats,
        indexed_count=indexed_count,
        skipped_count=skipped_count,
        results=results,
    )


@router.post(
    "/parse", response_model=ParsedPageResult, summary="Спарсить страницу для RAG"
)
async def parse_page_for_rag(url: HttpUrl = Query(..., description="URL страницы")):
    """Парсит страницу или PDF-документ, очищает текст через LLM и находит документы."""
    url_str = str(url)

    # 1. Проверяем тип контента (может это PDF документ)
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            resp = await client.head(url_str, follow_redirects=True)
            content_type = resp.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or url_str.lower().endswith(".pdf"):
                logger.info(
                    f"Обнаружен PDF по URL: {url_str}. Запуск Vision парсера для превью."
                )

                resp_get = await client.get(url_str, follow_redirects=True)
                resp_get.raise_for_status()

                pdf_markdown = get_abbrev_expander().expand(
                    await process_pdf_bytes(resp_get.content)
                )

                # Декодируем URL для fallback
                # (чтобы вместо %D0%98 было нормальное русское название)
                raw_filename = url_str.split("/")[-1]
                decoded_title = unquote(raw_filename)

                generated_title = await generate_title_from_text(pdf_markdown)
                title_to_use = (
                    generated_title
                    if generated_title and generated_title != "Без названия"
                    else decoded_title
                )

                # Добавляем сгенерированный заголовок в текст, если его там нет
                if title_to_use != decoded_title and not pdf_markdown.startswith(
                    f"# {title_to_use}"
                ):
                    pdf_markdown = f"# {title_to_use}\n\n{pdf_markdown}"

                return ParsedPageResult(
                    # Сгенерированное нейронкой или нормальное имя файла как заголовок
                    title=title_to_use,
                    url=url_str,
                    text=pdf_markdown,
                    documents=[],  # Внутри PDF ссылок на другие документы мы не собираем
                )
    except Exception as e:
        logger.error(f"Ошибка загрузки {url_str} перед превью: {e}")
        raise HTTPException(status_code=400, detail="Failed to fetch URL check") from e

    # 2. Если это обычная HTML страница
    # Добавляем стандартные заголовки, чтобы избежать блокировки (Requirement 1 - Fix)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    # Расширенный список стандартных селекторов для сайта НГУ и других
    default_selectors = [
        {"style": "div", "attr": "content-block"},
        {"style": "div", "attr": "content-bar"},
        {"style": "div", "attr": "content"},
        {"style": "div", "attr": "main-content"},
        {"style": "div", "attr": "post-content"},
        {"style": "article", "attr": ""},
        {"style": "main", "attr": ""},
    ]

    data = parse_page(str(url), selectors=default_selectors, headers=headers)
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Failed to parse page (possible block or invalid URL)",
        )

    # Собираем текст из блоков
    text_parts = []
    if data.get("title") and data.get("title") != "Не найдено":
        text_parts.append(f"# {data['title']}")

    if data.get("description"):
        text_parts.append(data["description"])

    content_blocks = data.get("content_blocks", [])

    # Дедупликация и очистка блоков (если один блок полностью содержит другой)
    cleaned_blocks = []
    if content_blocks:
        # Сортируем по длине (сначала самые длинные)
        sorted_blocks = sorted(list(set(content_blocks)), key=len, reverse=True)
        for block in sorted_blocks:
            # Если этот блок еще не является частью уже добавленного блока
            if not any(block in existing for existing in cleaned_blocks):
                cleaned_blocks.append(block)

    # Если блоков не нашли, пытаемся достать хоть что-то из body (Fallback)
    if not cleaned_blocks:
        try:
            resp = requests.get(str(url), headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Убираем скрипты и стили
                for script in soup(["script", "style"]):
                    script.extract()
                # Берем текст из body, если он там есть
                body = soup.body
                if body:
                    # Пытаемся взять основной текст, пропуская навигацию если возможно
                    body_text = body.get_text(separator="\n", strip=True)
                    if len(body_text) > 100:
                        cleaned_blocks = [body_text]
        except Exception:
            pass

    text_parts.extend(cleaned_blocks)
    raw_text = "\n\n".join(text_parts)

    # Очищаем через LLM preprocessor (Requirement 1)
    clean_text = await clean_and_structure_text(raw_text)

    # Генерируем заголовок с помощью LLM (так же, как для PDF)
    generated_title_html = await generate_title_from_text(clean_text)
    html_title_to_use = (
        generated_title_html
        if generated_title_html and generated_title_html != "Без названия"
        else data.get("title", "Не найдено")
    )

    # Преобразуем документы
    docs = [
        ParsedDocument(title=d["name"], url=d["url"]) for d in data.get("documents", [])
    ]

    return ParsedPageResult(
        title=html_title_to_use,
        url=str(url),
        text=clean_text,
        documents=docs,
    )


@router.post("/confirm", status_code=201, summary="Подтвердить загрузку в RAG")
async def confirm_rag_upload(request: ConfirmUploadRequest):
    """Загружает отредактированный текст и выбранные документы в RAG."""

    try:
        # 1. Загружаем текст страницы (если он есть и не пуст)
        if request.text and request.text.strip():
            prepared_text = get_abbrev_expander().expand(request.text.strip())
            await add_texts_async(
                texts=[prepared_text],
                graph_id=DEFAULT_GRAPH_ID,
                source_ids=[request.title],
                file_paths=[request.url],
            )

        # 2. Вложенные документы больше не загружаем (по запросу)
        # if request.documents:
        #     for doc in request.documents:
        #         logger.info(f"Парсинг вложенного документа: {doc.url}")
        #         await parse_and_save_url(doc.url, title=doc.title)

        return {"status": "success", "message": "Content and documents uploaded to RAG"}
    except Exception as e:
        logger.error(f"Error during RAG confirmation upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")


@router.get(
    "/docs", response_model=RagDocumentListResponse, summary="Список документов в RAG"
)
async def list_rag_documents():
    """Возвращает список всех загруженных в RAG документов."""
    memory = get_graph_memory()
    docs = await memory.get_list_docs(DEFAULT_GRAPH_ID)
    return RagDocumentListResponse(documents=docs)


@router.get(
    "/docs/{doc_id:path}/content",
    response_model=RagDocumentContentResponse,
    summary="Получить полный текст документа",
)
async def get_rag_document_content(doc_id: str):
    """Возвращает полный исходный текст документа по его ID."""
    memory = get_graph_memory()
    content = await memory.get_doc_full_text(DEFAULT_GRAPH_ID, doc_id)

    if content is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    return RagDocumentContentResponse(id=doc_id, content=content)


@router.delete(
    "/docs/{doc_id:path}",
    summary="Удалить документ из RAG",
)
async def delete_rag_document(doc_id: str):
    """Удаляет документ из базы знаний RAG."""
    memory = get_graph_memory()
    success = await memory.delete_doc(DEFAULT_GRAPH_ID, doc_id)
    if not success:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document '{doc_id}'"
        )

    return {"status": "success", "message": f"Document '{doc_id}' deleted."}


@router.post(
    "/upload/csv",
    response_model=CsvImportResponse,
    summary="Импорт документов по ссылкам из CSV",
)
async def upload_csv_documents(
    file: UploadFile = File(
        ..., description="CSV файл с полями (Название, Link, Комментарий)"
    ),
):
    """
    Принимает CSV документ на вход и обрабатывает все документы там.
    Ожидаемый формат: Название,Link,Комментарий
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")

    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Файл должен быть в кодировке UTF-8")

    reader = csv.DictReader(io.StringIO(text))
    # Проверяем наличие колонки Link
    if not reader.fieldnames or "Link" not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'Link'")

    results = []

    for row in reader:
        url = row.get("Link", "").strip()
        title = row.get("Название", "").strip()

        if not url:
            continue

        if not title:
            try:
                # Если названия нет, пытаемся достать его по ссылке
                path = url.lower()
                if "application/pdf" in path or path.endswith(".pdf"):
                    # Для документов берем имя файла из ссылки
                    raw_filename = url.split("/")[-1]
                    title = unquote(raw_filename)
                else:
                    # Для веб-страниц пытаемся получить <title>
                    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
                        resp = await client.get(url, follow_redirects=True)
                        if resp.status_code == 200:
                            soup = BeautifulSoup(resp.text, "html.parser")
                            if soup.title and soup.title.string:
                                title = soup.title.string.strip()
            except Exception as title_e:
                logger.warning(f"Не удалось извлечь название для {url}: {title_e}")

            # Если всё еще нет названия,
            # оставляем как есть (будет использован URL как fallback)
            if not title:
                title = url

        try:
            success = await parse_and_save_url(url=url, title=title)
            message = (
                "Успешно"
                if success
                else "Не удалось сохранить (возможно пустой контент или недоступен)"
            )
            results.append(
                CsvImportResult(title=title, url=url, success=success, message=message)
            )
        except Exception as e:
            logger.error(f"Ошибка при обработке {url} из CSV: {e}")
            results.append(
                CsvImportResult(title=title, url=url, success=False, message=str(e))
            )

    return CsvImportResponse(
        imported_count=sum(1 for r in results if r.success),
        total_found=len(results),
        results=results,
    )


@router.post(
    "/upload/csv/preview",
    response_model=CsvImportPreviewResponse,
    summary="Превью документов из CSV (без обработки)",
)
async def preview_csv_documents(
    file: UploadFile = File(
        ..., description="CSV файл с полями (Название, Link, Комментарий)"
    ),
):
    """
    Принимает CSV документ и возвращает список найденных в нём ссылок без их обработки.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")

    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Файл должен быть в кодировке UTF-8")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "Link" not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'Link'")

    results = []
    for row in reader:
        url = row.get("Link", "").strip()
        title = row.get("Название", "").strip()
        comment = row.get("Комментарий", "").strip()

        if not url:
            continue

        if not title:
            # Fallback to URL or filename from URL
            raw_filename = url.split("/")[-1]
            title = unquote(raw_filename) if raw_filename else url

        results.append(CsvImportPreviewResult(title=title, url=url, comment=comment))

    return CsvImportPreviewResponse(total_found=len(results), results=results)
