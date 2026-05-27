import hashlib
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from abbrev.expander import get_abbrev_expander
from db.postgres.models import Document
from db.postgres.services.document import DocumentService
from parser.url import process_url
from rag.graph_memory import get_graph_memory
from rag.loader import DEFAULT_GRAPH_ID, add_texts_async

logger = logging.getLogger(__name__)


class DocumentUpdateService:
    def __init__(self, session: AsyncSession, graph_id: str = DEFAULT_GRAPH_ID):
        self.session = session
        self.graph_id = graph_id
        self.documents = DocumentService(session)

    async def check_documents(
        self, document_ids: list[str] | None = None
    ) -> list[dict]:
        docs = await self.documents.list_active(self.graph_id, document_ids)
        results = []
        for document in docs:
            results.append(await self._check_document(document))
        return results

    async def update_changed_documents(
        self, document_ids: list[str] | None = None
    ) -> list[dict]:
        checks = await self.check_documents(document_ids)
        results = []
        for check in checks:
            if check["status"] != "changed":
                results.append(check)
                continue

            document = await self.documents.get_by_id(check["id"])
            if document is None:
                results.append({**check, "status": "failed", "message": "Not found"})
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
                        "status": "failed",
                        "message": str(exc),
                    }
                )
        return results

    async def _check_document(self, document: Document) -> dict:
        if not document.source_url:
            return {
                "id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "status": "skipped",
                "content_hash": document.content_hash,
                "previous_hash": document.content_hash,
                "message": "Document has no source_url",
            }

        try:
            raw = await fetch_url_bytes(document.source_url)
            content_hash = calculate_content_hash(raw)
        except Exception as exc:
            return {
                "id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "status": "failed",
                "content_hash": None,
                "previous_hash": document.content_hash,
                "message": str(exc),
            }

        status = (
            "changed"
            if document.content_hash is None or document.content_hash != content_hash
            else "unchanged"
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
                "status": "skipped",
                "content_hash": document.content_hash,
                "previous_hash": document.content_hash,
                "message": "Document has no source_url",
            }

        text = await process_url(document.source_url)
        if not text:
            raise ValueError("Could not parse document content")

        prepared_text = get_abbrev_expander().expand(text.strip())
        if not prepared_text:
            raise ValueError("Parsed document content is empty")

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
            raise ValueError("Failed to save updated document to RAG")

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
            "status": "updated",
            "content_hash": content_hash,
            "previous_hash": previous_hash,
            "message": None,
        }


async def fetch_url_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content


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
