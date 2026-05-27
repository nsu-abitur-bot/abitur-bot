import asyncio
import hashlib
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from abbrev.expander import get_abbrev_expander
from db.postgres.models import Document
from db.postgres.services.document import (
    DOCUMENT_STATUS_CHANGED,
    DOCUMENT_STATUS_INDEXING_FAILED,
    DOCUMENT_STATUS_NO_SOURCE,
    DOCUMENT_STATUS_SOURCE_UNAVAILABLE,
    DOCUMENT_STATUS_UNCHANGED,
    DOCUMENT_STATUS_UPDATED,
    DocumentService,
)
from parser.url import process_url
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

logger = logging.getLogger(__name__)


class DocumentUpdateService:
    def __init__(
        self,
        session: AsyncSession,
        graph_id: str = DEFAULT_GRAPH_ID,
        check_concurrency: int = 8,
    ):
        self.session = session
        self.graph_id = graph_id
        self.check_concurrency = check_concurrency
        self.documents = DocumentService(session)

    async def check_documents(
        self, document_ids: list[str] | None = None
    ) -> list[dict]:
        docs = await self.documents.list_checkable(self.graph_id, document_ids)
        semaphore = asyncio.Semaphore(self.check_concurrency)
        results = await asyncio.gather(
            *(self._check_document_limited(document, semaphore) for document in docs)
        )
        await self.documents.save_check_results(results)
        return results

    async def update_changed_documents(
        self, document_ids: list[str] | None = None
    ) -> list[dict]:
        checks = await self.check_documents(document_ids)
        results = []
        for check in checks:
            if check["status"] != DOCUMENT_STATUS_CHANGED:
                results.append(check)
                continue

            document = await self.documents.get_by_id(check["id"])
            if document is None:
                results.append(
                    {
                        **check,
                        "status": DOCUMENT_STATUS_INDEXING_FAILED,
                        "message": "Документ не найден в базе",
                    }
                )
                continue

            try:
                updated = await self._update_document(document, check["content_hash"])
                results.append(updated)
            except Exception as exc:
                logger.exception("Failed to update document %s", document.id)
                await self.documents.mark_error(document.id)
                results.append(
                    {
                        **check,
                        "status": DOCUMENT_STATUS_INDEXING_FAILED,
                        "message": str(exc),
                    }
                )
        return results

    async def _check_document_limited(
        self, document: Document, semaphore: asyncio.Semaphore
    ) -> dict:
        async with semaphore:
            return await self._check_document(document)

    async def _check_document(self, document: Document) -> dict:
        if not document.source_url:
            return {
                "id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "status": DOCUMENT_STATUS_NO_SOURCE,
                "content_hash": document.content_hash,
                "previous_hash": document.content_hash,
                "message": "У документа не указана ссылка на источник",
            }

        try:
            raw = await fetch_url_bytes(document.source_url)
            content_hash = calculate_content_hash(raw)
        except Exception as exc:
            message = readable_fetch_error(exc)
            return {
                "id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "status": DOCUMENT_STATUS_SOURCE_UNAVAILABLE,
                "content_hash": None,
                "previous_hash": document.content_hash,
                "message": message,
            }

        status = (
            DOCUMENT_STATUS_CHANGED
            if document.content_hash is None or document.content_hash != content_hash
            else DOCUMENT_STATUS_UNCHANGED
        )
        return {
            "id": document.id,
            "title": document.title,
            "source_url": document.source_url,
            "status": status,
            "content_hash": content_hash,
            "previous_hash": document.content_hash,
            "message": None,
        }

    async def _update_document(self, document: Document, content_hash: str) -> dict:
        if not document.source_url:
            return {
                "id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "status": DOCUMENT_STATUS_NO_SOURCE,
                "content_hash": document.content_hash,
                "previous_hash": document.content_hash,
                "message": "У документа не указана ссылка на источник",
            }

        text = await process_url(document.source_url)
        if not text:
            raise ValueError("Не удалось извлечь текст из документа")

        prepared_text = get_abbrev_expander().expand(text.strip())
        if not prepared_text:
            raise ValueError("После обработки текст документа пустой")

        previous_hash = document.content_hash
        old_rag_doc_id = document.rag_doc_id
        memory = get_graph_memory()
        await memory.delete_doc(self.graph_id, old_rag_doc_id)
        saved_count = await add_texts_async(
            texts=[prepared_text],
            graph_id=self.graph_id,
            source_ids=[document.id],
            file_paths=[document.source_url],
        )
        if saved_count == 0:
            raise ValueError("Не удалось сохранить обновленный документ в RAG")

        updated = await self.documents.mark_indexed(
            document.id,
            content_hash=content_hash,
            content_length=len(prepared_text),
            rag_doc_id=document.id,
        )
        return {
            "id": document.id,
            "title": updated.title if updated else document.title,
            "source_url": document.source_url,
            "status": DOCUMENT_STATUS_UPDATED,
            "content_hash": content_hash,
            "previous_hash": previous_hash,
            "message": None,
        }


async def fetch_url_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content


def readable_fetch_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 404:
            return "Источник не найден: сервер вернул 404"
        if status_code == 403:
            return "Источник недоступен: сервер запретил доступ (403)"
        if status_code == 401:
            return "Источник требует авторизацию (401)"
        if status_code >= 500:
            return f"Источник временно недоступен: ошибка сервера {status_code}"
        return f"Источник недоступен: HTTP {status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "Источник не ответил вовремя"
    if isinstance(exc, httpx.ConnectError):
        return "Не удалось подключиться к источнику"
    if isinstance(exc, httpx.InvalidURL):
        return "Некорректная ссылка на источник"
    return "Не удалось проверить источник"


def calculate_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def first_http_url(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(","):
        candidate = part.strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
    return None
