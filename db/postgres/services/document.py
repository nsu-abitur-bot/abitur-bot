from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Document, timestamp


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
            Document.status != "deleted",
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active(
        self, graph_id: str, document_ids: Sequence[str] | None = None
    ) -> Sequence[Document]:
        stmt = select(Document).where(
            Document.graph_id == graph_id,
            Document.status == "active",
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
                status="active",
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
            document.status = "active"
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
        document.status = "active"
        document.last_indexed_at = timestamp()
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def mark_error(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = "error"
        await self.session.commit()
        return True

    async def mark_deleted(self, document_id: str) -> bool:
        document = await self.get_by_id(document_id)
        if document is None:
            return False
        document.status = "deleted"
        await self.session.commit()
        return True
