from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from uuid6 import uuid7

from config.content_gate import content_write
from db.postgres.models import Document, timestamp

DOCUMENT_STATUS_INDEXED = "проиндексирован"
DOCUMENT_STATUS_UNCHANGED = "без изменений"
DOCUMENT_STATUS_CHANGED = "изменён"
DOCUMENT_STATUS_SOURCE_UNAVAILABLE = "источник недоступен"
DOCUMENT_STATUS_NO_SOURCE = "нет ссылки"
DOCUMENT_STATUS_INDEXING_FAILED = "ошибка индексации"
DOCUMENT_STATUS_DELETED = "удалён"
DOCUMENT_STATUS_UPDATED = "обновлён"
SOURCE_URL_UNIQUE_INDEX = "ix_document_graph_source_url_active"


class DocumentSourceUrlConflict(ValueError):
    """The URL already belongs to another active document in this graph."""


def _is_source_url_unique_violation(exc: IntegrityError) -> bool:
    cause = getattr(exc.orig, "__cause__", None)
    diagnostic = getattr(exc.orig, "diag", None)
    return SOURCE_URL_UNIQUE_INDEX in {
        getattr(exc.orig, "constraint_name", None),
        getattr(cause, "constraint_name", None),
        getattr(diagnostic, "constraint_name", None),
    }


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

    async def get_by_ids(self, document_ids: Sequence[str]) -> Sequence[Document]:
        """Документы одним IN-запросом: чекер зовёт это для сотен строк (#300)."""
        ids = list(dict.fromkeys(document_ids))
        if not ids:
            return []
        stmt = select(Document).where(Document.id.in_(ids))
        return (await self.session.execute(stmt)).scalars().all()

    async def get_by_rag_doc_id(
        self, graph_id: str, rag_doc_id: str
    ) -> Optional[Document]:
        stmt = select(Document).where(
            Document.graph_id == graph_id,
            Document.rag_doc_id == rag_doc_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_source_proposal_id(self, proposal_id: str) -> Optional[Document]:
        result = await self.session.execute(
            select(Document).where(Document.source_proposal_id == proposal_id)
        )
        return result.scalar_one_or_none()

    async def get_by_source_url(
        self, graph_id: str, source_url: str
    ) -> Optional[Document]:
        stmt = (
            select(Document)
            .where(
                Document.graph_id == graph_id,
                Document.source_url == source_url,
                Document.status.not_in({"deleted", DOCUMENT_STATUS_DELETED}),
            )
            .order_by(Document.updated_at.desc(), Document.id.desc())
            .limit(1)
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

    @content_write()
    async def create_or_update_for_source(
        self,
        *,
        graph_id: str,
        title: str,
        source_url: str | None,
        content_hash: str | None = None,
        content_length: int | None = None,
        rag_doc_id: str | None = None,
        source_proposal_id: str | None = None,
    ) -> Document:
        document = (
            await self.get_by_source_url(graph_id, source_url) if source_url else None
        )

        if document is None and rag_doc_id:
            document = await self.get_by_rag_doc_id(graph_id, rag_doc_id)

        created = False
        if document is None:
            document_id = str(uuid7())
            document = Document(
                id=document_id,
                title=title,
                source_url=source_url,
                graph_id=graph_id,
                rag_doc_id=rag_doc_id or document_id,
                source_proposal_id=source_proposal_id,
                content_hash=content_hash,
                content_length=content_length,
                status="pending" if source_proposal_id else DOCUMENT_STATUS_INDEXED,
                last_indexed_at=None if source_proposal_id else timestamp(),
            )
            try:
                async with self.session.begin_nested():
                    self.session.add(document)
                    await self.session.flush()
                created = True
            except IntegrityError as exc:
                if not source_url or not _is_source_url_unique_violation(exc):
                    raise
                document = await self.get_by_source_url(graph_id, source_url)
                if document is None:
                    raise

        if not created:
            document.title = title
            if source_url is not None:
                document.source_url = source_url
            if content_hash is not None:
                document.content_hash = content_hash
            if content_length is not None:
                document.content_length = content_length
            if not document.rag_doc_id:
                document.rag_doc_id = document.id

        await self.session.commit()
        await self.session.refresh(document)
        return document

    @content_write()
    async def mark_indexed(
        self,
        document_id: str,
        *,
        content_hash: str | None,
        content_length: int | None,
        title: str | None = None,
        source_url: str | None = None,
        rag_doc_id: str | None = None,
        source_proposal_id: str | None = None,
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
        if source_proposal_id is not None:
            document.source_proposal_id = source_proposal_id
        document.content_hash = content_hash
        document.content_length = content_length
        document.status = DOCUMENT_STATUS_INDEXED
        document.last_checked_hash = content_hash
        document.last_check_message = None
        document.last_indexed_at = timestamp()
        await self.session.commit()
        await self.session.refresh(document)
        return document

    @content_write()
    async def mark_error(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = DOCUMENT_STATUS_INDEXING_FAILED
        document.last_check_message = "Не удалось проиндексировать документ"
        await self.session.commit()
        return True

    @content_write()
    async def mark_deleted(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = DOCUMENT_STATUS_DELETED
        await self.session.commit()
        return True

    @content_write()
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
        if source_url_changed and source_url:
            existing = await self.get_by_source_url(document.graph_id, source_url)
            if existing is not None and existing.id != document.id:
                raise DocumentSourceUrlConflict(source_url)
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

        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            if _is_source_url_unique_violation(exc):
                raise DocumentSourceUrlConflict(source_url or "") from exc
            raise
        await self.session.refresh(document)
        return document

    @content_write()
    async def save_check_results(self, results: Sequence[dict]) -> None:
        checked_at = timestamp()
        # Один IN-запрос вместо SELECT на каждый результат (#300).
        documents = {d.id: d for d in await self.get_by_ids([r["id"] for r in results])}
        for result in results:
            document = documents.get(result["id"])
            if document is None:
                continue
            document_status = CHECK_TO_DOCUMENT_STATUS.get(result["status"])
            if document_status is not None:
                document.status = document_status
            document.last_checked_at = checked_at
            document.last_checked_hash = result.get("content_hash")
            document.last_check_message = result.get("message")

        await self.session.commit()
