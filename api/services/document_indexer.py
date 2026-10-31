import asyncio
import logging
import os
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document
from db.postgres.services.document import DocumentService
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

logger = logging.getLogger(__name__)

# Статусы, при которых новая версия считается успешно проиндексированной.
_READY_STATUSES: frozenset[str] = frozenset({"processed", "preprocessed"})
# Статус окончательного сбоя — переключение запрещено.
_FAILED_STATUS = "failed"


async def _wait_for_doc_ready(
    graph_id: str,
    new_rag_doc_id: str,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
) -> bool:
    """Ожидает перехода нового документа в статус PROCESSED/PREPROCESSED.

    Причины ожидания и строгой проверки:
    - saved_count > 0 не гарантирует успешности: GraphMemory.save() возвращает True,
      если rag.ainsert() завершился, но LightRAG перехватывает сбои extraction/merge
      и сохраняет DocStatus.FAILED без выбрасывания исключения.
    - При занятом pipeline обработка лишь ставится в очередь (DocStatus.PENDING),
      а ainsert() уже возвращается.
    В обоих случаях переключение Postgres и удаление старой версии недопустимы,
    пока не подтверждён успешный PROCESSED именно для new_rag_doc_id.
    """
    memory = get_graph_memory()
    if not hasattr(memory, "get_doc_status"):
        return True

    start_time = time.monotonic()
    while True:
        status = await memory.get_doc_status(graph_id, new_rag_doc_id)
        if status in _READY_STATUSES:
            return True

        if status == _FAILED_STATUS:
            logger.error(
                "Новая версия документа %s в графе %s помечена как FAILED в LightRAG;"
                " переключение отменено, старая версия сохранена",
                new_rag_doc_id,
                graph_id,
            )
            return False

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout:
            logger.warning(
                "Новая версия документа %s в графе %s не достигла PROCESSED за %.2fс"
                " (status=%r); переключение отменено, старая версия сохранена",
                new_rag_doc_id,
                graph_id,
                elapsed,
                status,
            )
            return False

        await asyncio.sleep(min(poll_interval, max(0.001, timeout - elapsed)))


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
    wait_timeout: Optional[float] = None,
) -> Optional[Document]:
    """Индексирует документ в RAG с безопасной заменой версии (add → switch → delete).

    Порядок:
    1. ADD: индексируем новую версию под новым rag_doc_id.
    2. VERIFY: ожидаем статус PROCESSED именно для new_rag_doc_id.
       При FAILED или незавершённой обработке (PENDING/таймаут очереди)
       старая версия сохраняется, переключение в Postgres не выполняется.
    3. SWITCH: при подтверждённом PROCESSED атомарно переключаем
       Document.rag_doc_id в Postgres.
    4. DELETE: удаляем старую версию из LightRAG через delete_doc_safe,
       сохраняя чанки, общие с новой версией (chunk_id = hash(text)).

    При любом сбое на шагах 1–2 старая версия документа остаётся нетронутой.
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
                    wait_timeout=wait_timeout,
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

    # 2. VERIFY: дожидаемся подтверждённого статуса PROCESSED для new_rag_doc_id
    resolved_wait_timeout = (
        wait_timeout
        if wait_timeout is not None
        else float(os.getenv("RAG_INDEX_WAIT_TIMEOUT", "30.0"))
    )
    new_doc_ready = await _wait_for_doc_ready(
        graph_id,
        new_rag_doc_id,
        timeout=resolved_wait_timeout,
    )
    if not new_doc_ready:
        if hasattr(document_service, "mark_error"):
            await document_service.mark_error(document.id)
        return None

    # 3. SWITCH: атомарно переключаем Document.rag_doc_id и фиксируем в БД
    updated_doc = await document_service.mark_indexed(
        document.id,
        content_hash=content_hash,
        content_length=len(text),
        title=resolved_title,
        source_url=resolved_source_url,
        rag_doc_id=new_rag_doc_id,
    )

    # 4. DELETE: удаляем старую версию документа из RAG, сохраняя общие чанки
    if old_rag_doc_id and old_rag_doc_id != new_rag_doc_id:
        try:
            memory = get_graph_memory()
            if hasattr(memory, "delete_doc_safe"):
                new_chunks = (
                    await memory.get_chunks_list(graph_id, new_rag_doc_id)
                    if hasattr(memory, "get_chunks_list")
                    else []
                )
                await memory.delete_doc_safe(
                    graph_id,
                    old_rag_doc_id,
                    preserve_chunk_ids=set(new_chunks),
                )
            else:
                await memory.delete_doc(graph_id, old_rag_doc_id)
        except Exception as exc:
            logger.warning(
                "Не удалось удалить старую версию документа %s (rag_doc_id=%s): %s",
                document.id,
                old_rag_doc_id,
                exc,
            )

    return updated_doc or document
