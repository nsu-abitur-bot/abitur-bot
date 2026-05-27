from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Document, timestamp

DOCUMENT_STATUS_INDEXED = "проиндексирован"
DOCUMENT_STATUS_UNCHANGED = "без изменений"
DOCUMENT_STATUS_CHANGED = "изменён"
DOCUMENT_STATUS_SOURCE_UNAVAILABLE = "источник недоступен"
DOCUMENT_STATUS_NO_SOURCE = "нет ссылки"
DOCUMENT_STATUS_INDEXING_FAILED = "ошибка индексации"
DOCUMENT_STATUS_DELETED = "удалён"
DOCUMENT_STATUS_UPDATED = "обновлён"

CHECK_TO_DOCUMENT_STATUS = {
    DOCUMENT_STATUS_UNCHANGED: DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_CHANGED: DOCUMENT_STATUS_CHANGED,
    DOCUMENT_STATUS_SOURCE_UNAVAILABLE: DOCUMENT_STATUS_SOURCE_UNAVAILABLE,
    DOCUMENT_STATUS_NO_SOURCE: DOCUMENT_STATUS_NO_SOURCE,
    DOCUMENT_STATUS_UPDATED: DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_INDEXING_FAILED: DOCUMENT_STATUS_INDEXING_FAILED,
}

ACTIVE_DOCUMENT_STATUSES = {
    "active",
    "indexed",
    "changed",
    "source_unavailable",
    "no_source",
    "indexing_failed",
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_CHANGED,
    DOCUMENT_STATUS_SOURCE_UNAVAILABLE,
    DOCUMENT_STATUS_NO_SOURCE,
    DOCUMENT_STATUS_INDEXING_FAILED,
}


class DocumentService:
    """Сервис для документов, проиндексированных в RAG."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, document_id: str) -> Optional[Document]:
        stmt = select(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_rag_doc_id(
        self, graph_id: str, rag_doc_id: str
    ) -> Optional[Document]:
        stmt = select(Document).where(
            Document.graph_id == graph_id,
            Document.rag_doc_id == rag_doc_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_source_url(
        self, graph_id: str, source_url: str
    ) -> Optional[Document]:
        stmt = select(Document).where(
            Document.graph_id == graph_id,
            Document.source_url == source_url,
            Document.status.not_in({"deleted", DOCUMENT_STATUS_DELETED}),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active(
        self, graph_id: str, document_ids: Sequence[str] | None = None
    ) -> Sequence[Document]:
        return await self.list_checkable(graph_id, document_ids)

    async def list_checkable(
        self, graph_id: str, document_ids: Sequence[str] | None = None
    ) -> Sequence[Document]:
        stmt = select(Document).where(
            Document.graph_id == graph_id,
            Document.status.in_(ACTIVE_DOCUMENT_STATUSES),
        )
        if document_ids is not None:
            stmt = stmt.where(Document.id.in_(document_ids))
        stmt = stmt.order_by(Document.updated_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create_or_update_for_source(
        self,
        *,
        graph_id: str,
        title: str,
        source_url: str | None,
        content_hash: str | None = None,
        content_length: int | None = None,
        rag_doc_id: str | None = None,
    ) -> Document:
        document = (
            await self.get_by_source_url(graph_id, source_url) if source_url else None
        )

        if document is None and rag_doc_id:
            document = await self.get_by_rag_doc_id(graph_id, rag_doc_id)

        if document is None:
            document = Document(
                title=title,
                source_url=source_url,
                graph_id=graph_id,
                rag_doc_id=rag_doc_id or "",
                content_hash=content_hash,
                content_length=content_length,
                status=DOCUMENT_STATUS_INDEXED,
                last_indexed_at=timestamp(),
            )
            self.session.add(document)
            await self.session.flush()
            if not document.rag_doc_id:
                document.rag_doc_id = document.id
        else:
            document.title = title
            document.source_url = source_url
            document.content_hash = content_hash
            document.content_length = content_length
            document.status = DOCUMENT_STATUS_INDEXED
            document.last_indexed_at = timestamp()
            if not document.rag_doc_id:
                document.rag_doc_id = document.id

        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def mark_indexed(
        self,
        document_id: str,
        *,
        content_hash: str | None,
        content_length: int | None,
        title: str | None = None,
        source_url: str | None = None,
        rag_doc_id: str | None = None,
    ) -> Optional[Document]:
        document = await self.get_by_id(document_id)
        if document is None:
            return None
        if title is not None:
            document.title = title
        if source_url is not None:
            document.source_url = source_url
        if rag_doc_id is not None:
            document.rag_doc_id = rag_doc_id
        document.content_hash = content_hash
        document.content_length = content_length
        document.status = DOCUMENT_STATUS_INDEXED
        document.last_checked_hash = content_hash
        document.last_check_message = None
        document.last_indexed_at = timestamp()
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def mark_error(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = DOCUMENT_STATUS_INDEXING_FAILED
        document.last_check_message = "Не удалось проиндексировать документ"
        await self.session.commit()
        return True

    async def mark_deleted(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = DOCUMENT_STATUS_DELETED
        await self.session.commit()
        return True

    async def update_metadata(
        self,
        document_id: str,
        *,
        title: str | None = None,
        source_url: str | None = None,
    ) -> Optional[Document]:
        document = await self.get_by_id(document_id)
        if document is None:
            return None

        source_url_changed = source_url is not None and source_url != document.source_url
        if title is not None:
            document.title = title
        if source_url is not None:
            document.source_url = source_url
        if source_url_changed:
            document.status = (
                DOCUMENT_STATUS_CHANGED if source_url else DOCUMENT_STATUS_NO_SOURCE
            )
            document.last_checked_hash = None
            document.last_check_message = "Ссылка на источник изменена"

        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def save_check_results(self, results: Sequence[dict]) -> None:
        checked_at = timestamp()
        for result in results:
            document_id = result["id"]
            document = await self.get_by_id(document_id)
            if document is None:
                continue
            document_status = CHECK_TO_DOCUMENT_STATUS.get(result["status"])
            if document_status is not None:
                document.status = document_status
            document.last_checked_at = checked_at
            document.last_checked_hash = result.get("content_hash")
            document.last_check_message = result.get("message")

        await self.session.commit()
