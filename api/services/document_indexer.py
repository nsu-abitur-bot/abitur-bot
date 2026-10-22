import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document
from db.postgres.services.document import DocumentService
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

logger = logging.getLogger(__name__)


async def index_document(
    *,
    text: str,
    document: Optional[Document] = None,
    title: Optional[str] = None,
    source_url: Optional[str] = None,
    content_hash: Optional[str] = None,
    file_path: Optional[str] = None,
    graph_id: str = DEFAULT_GRAPH_ID,
    session: Optional[AsyncSession] = None,
    document_service: Optional[DocumentService] = None,
) -> Optional[Document]:
    """Индексирует документ в RAG с безопасной заменой версии (add -> switch -> delete).

    Порядок:
    1. ADD: индексируем новую версию под новым rag_doc_id.
    2. SWITCH: при успехе атомарно переключаем Document.rag_doc_id на новый ID в Postgres.
    3. DELETE: удаляем старую версию (old_rag_doc_id) из LightRAG.

    При сбое индексации старая версия документа остаётся нетронутой в RAG
    и доступной боту.
    """
    if document_service is None:
        if session is not None:
            document_service = DocumentService(session)
        else:
            async with AsyncSessionLocal() as auto_session:
                return await index_document(
                    text=text,
                    document=document,
                    title=title,
                    source_url=source_url,
                    content_hash=content_hash,
                    file_path=file_path,
                    graph_id=graph_id,
                    session=auto_session,
                )

    resolved_title = (
        title
        or (getattr(document, "title", None) if document else None)
        or file_path
        or source_url
        or "Документ"
    )
    resolved_source_url = (
        source_url
        if source_url is not None
        else (getattr(document, "source_url", None) if document else None)
    )

    if document is None:
        document = await document_service.create_or_update_for_source(
            graph_id=graph_id,
            title=resolved_title,
            source_url=resolved_source_url,
        )

    old_rag_doc_id = document.rag_doc_id
    new_rag_doc_id = str(uuid7())
    resolved_file_path = file_path or resolved_source_url or resolved_title

    # 1. ADD: индексируем новую версию под новым rag_doc_id
    try:
        saved_count = await add_texts_async(
            texts=[text],
            graph_id=graph_id,
            source_ids=[new_rag_doc_id],
            file_paths=[resolved_file_path] if resolved_file_path else None,
        )
    except Exception as exc:
        logger.exception(
            "Ошибка при индексации документа %s в RAG (graph %s): %s",
            getattr(document, "id", None),
            graph_id,
            exc,
        )
        if hasattr(document_service, "mark_error"):
            await document_service.mark_error(document.id)
        return None

    if saved_count == 0:
        logger.warning(
            "Не удалось сохранить документ %s в RAG (saved_count == 0)",
            getattr(document, "id", None),
        )
        if hasattr(document_service, "mark_error"):
            await document_service.mark_error(document.id)
        return None

    # 2. SWITCH: атомарно переключаем Document.rag_doc_id и фиксируем в БД
    updated_doc = await document_service.mark_indexed(
        document.id,
        content_hash=content_hash,
        content_length=len(text),
        title=resolved_title,
        source_url=resolved_source_url,
        rag_doc_id=new_rag_doc_id,
    )

    # 3. DELETE: удаляем старую версию документа из RAG
    if old_rag_doc_id and old_rag_doc_id != new_rag_doc_id:
        try:
            memory = get_graph_memory()
            await memory.delete_doc(graph_id, old_rag_doc_id)
        except Exception as exc:
            logger.warning(
                "Не удалось удалить старую версию документа %s (rag_doc_id=%s): %s",
                document.id,
                old_rag_doc_id,
                exc,
            )

    return updated_doc or document
