import logging

import httpx
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import HttpUrl

from api.schemas.rag import (
    ConfirmUploadRequest,
    ParsedDocument,
    ParsedPageResult,
    RagDocumentContentResponse,
    RagDocumentListResponse,
    RagUploadResponse,
)
from api.services.rag_upload import RagUploadService
from llm.preprocessor import clean_and_structure_text
from parser.nsu_parser import parse_page
from parser.parser_to_rag import parse_and_save_url
from parser.url_parser import get_content_type, process_pdf_bytes
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
    content_type = await get_content_type(url_str)

    if "application/pdf" in content_type or url_str.lower().endswith(".pdf"):
        logger.info(
            f"Обнаружен PDF по URL: {url_str}. Запуск Vision парсера для превью."
        )
        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                resp = await client.get(url_str, follow_redirects=True)
                resp.raise_for_status()

            pdf_markdown = await process_pdf_bytes(resp.content)

            return ParsedPageResult(
                title=url_str.split("/")[-1],  # Берем имя файла как заголовок
                url=url_str,
                text=pdf_markdown,
                documents=[],  # Внутри PDF ссылок на другие документы мы не собираем
            )
        except Exception as e:
            logger.error(f"Ошибка при парсинге PDF для превью: {e}")
            raise HTTPException(
                status_code=400, detail=f"Failed to parse PDF file: {e}"
            )

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

    # Если LLM вернула сообщение об отсутствии текста (как у пользователя),
    # а у нас в raw_text что-то есть - значит LLM запуталась или raw_text плохой.
    # Но обычно clean_and_structure_text справляется.

    # Преобразуем документы
    docs = [
        ParsedDocument(title=d["name"], url=d["url"]) for d in data.get("documents", [])
    ]

    return ParsedPageResult(
        title=data.get("title", "Не найдено"),
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
            await add_texts_async(
                texts=[request.text],
                graph_id=DEFAULT_GRAPH_ID,
                source_ids=[request.title],
                file_paths=[request.url],
            )

        # 2. Обрабатываем прикреплённые документы (извлекаются из страницы на этапе /parse)
        if request.documents:
            for doc in request.documents:
                logger.info(f"Парсинг вложенного документа: {doc.url}")
                await parse_and_save_url(doc.url, title=doc.title)

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
