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
from llm.pdf_parser import parse_pdf_with_llm
from llm.preprocessor import clean_and_structure_text
from parser.nsu_parser import parse_page
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

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
    """Парсит страницу, очищает текст через LLM и находит документы."""
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
        import requests
        from bs4 import BeautifulSoup

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
    # 1. Загружаем текст страницы (если он есть и не пуст)
    try:
        if request.text and request.text.strip():
            await add_texts_async(
                texts=[request.text],
                graph_id=DEFAULT_GRAPH_ID,
                source_ids=[request.title],
                file_paths=[request.url],
            )

        # 2. Если есть прикреплённые документы, скачиваем, парсим LLM и загружаем
        if request.documents:
            import httpx

            # Для каждого документа скачиваем его PDF и парсим
            async with httpx.AsyncClient(timeout=120.0) as client:
                for doc in request.documents:
                    doc_title = doc.title
                    doc_url = doc.url
                    # Базовая валидация, нам нужны только PDF
                    if not doc_url.lower().endswith(".pdf"):
                        continue

                    try:
                        # 1) Скачивание
                        resp = await client.get(doc_url, follow_redirects=True)
                        if resp.status_code != 200:
                            import logging

                            logging.getLogger(__name__).warning(
                                f"Не удалось скачать: {doc_url}, статус: {resp.status_code}"
                            )
                            continue

                        pdf_bytes = resp.content

                        # 2) Парсинг PDF через LLM
                        parsed_md = await parse_pdf_with_llm(pdf_bytes)
                        if not parsed_md.strip():
                            continue

                        # 3) Закидываем в граф RAG
                        await add_texts_async(
                            texts=[parsed_md],
                            graph_id=DEFAULT_GRAPH_ID,
                            source_ids=[doc_title],
                            file_paths=[doc_url],
                        )
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).error(
                            f"Ошибка обработки документа {doc_title} ({doc_url}): {e}"
                        )

        return {"status": "success", "message": "Content and documents uploaded to RAG"}
    except Exception as e:
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
