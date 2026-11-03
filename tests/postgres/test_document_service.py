import asyncio
from datetime import datetime

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.routes.rag import update_rag_document
from api.schemas.rag import RagDocumentUpdateRequest
from db.postgres.models import Document
from db.postgres.services.document import (
    DOCUMENT_STATUS_CHANGED,
    DOCUMENT_STATUS_INDEXED,
    DOCUMENT_STATUS_UNCHANGED,
    DocumentService,
)


@pytest.mark.asyncio
async def test_save_check_results_preserves_document_statuses(session: AsyncSession):
    service = DocumentService(session)
    first = await service.create_or_update_for_source(
        graph_id="graph", title="doc-1", source_url="https://example.com/1"
    )
    second = await service.create_or_update_for_source(
        graph_id="graph", title="doc-2", source_url="https://example.com/2"
    )

    await service.save_check_results(
        [
            {
                "id": first.id,
                "status": DOCUMENT_STATUS_CHANGED,
                "content_hash": "hash-1",
                "message": "changed",
            },
            {
                "id": second.id,
                "status": DOCUMENT_STATUS_UNCHANGED,
                "content_hash": "hash-2",
                "message": None,
            },
        ]
    )

    first_after = await service.get_by_id(first.id)
    second_after = await service.get_by_id(second.id)
    assert first_after is not None and first_after.status == DOCUMENT_STATUS_CHANGED
    assert second_after is not None and second_after.status == DOCUMENT_STATUS_INDEXED


@pytest.mark.asyncio
async def test_get_by_source_url_chooses_latest_existing_duplicate(session: AsyncSession):
    await session.execute(
        text("DROP INDEX IF EXISTS ix_document_graph_source_url_active")
    )
    session.add_all(
        [
            Document(
                title=title,
                graph_id="graph",
                source_url="https://example.com/shared",
                rag_doc_id=rag_id,
                status=status,
                updated_at=updated_at,
            )
            for title, rag_id, status, updated_at in (
                ("old", "old", DOCUMENT_STATUS_INDEXED, datetime(2026, 1, 1)),
                ("new", "new", DOCUMENT_STATUS_INDEXED, datetime(2026, 1, 2)),
                ("deleted", "deleted", "удалён", datetime(2026, 1, 3)),
            )
        ]
    )
    await session.commit()

    found = await DocumentService(session).get_by_source_url(
        "graph", "https://example.com/shared"
    )

    assert found is not None and found.title == "new"


@pytest.mark.asyncio
async def test_update_metadata_rejects_occupied_url_with_http_409(session: AsyncSession):
    service = DocumentService(session)
    first = await service.create_or_update_for_source(
        graph_id="graph", title="first", source_url="https://example.com/first"
    )
    second = await service.create_or_update_for_source(
        graph_id="graph", title="second", source_url="https://example.com/second"
    )

    with pytest.raises(HTTPException) as error:
        await update_rag_document(
            second.id,
            RagDocumentUpdateRequest(source_url=first.source_url),
            session,
        )

    assert error.value.status_code == 409
    await session.refresh(second)
    assert second.source_url == "https://example.com/second"


@pytest.mark.asyncio
async def test_active_source_url_is_unique_but_deleted_url_can_be_reused(
    session: AsyncSession,
):
    service = DocumentService(session)
    first = await service.create_or_update_for_source(
        graph_id="graph", title="first", source_url="https://example.com/shared"
    )
    first_id = first.id

    session.add(
        Document(
            title="duplicate",
            graph_id="graph",
            source_url="https://example.com/shared",
            rag_doc_id="other-rag-id",
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()

    assert await service.mark_deleted(first_id)
    replacement = await service.create_or_update_for_source(
        graph_id="graph", title="replacement", source_url="https://example.com/shared"
    )
    assert replacement.id != first_id


@pytest.mark.asyncio
async def test_concurrent_create_for_same_url_reuses_document(test_engine: AsyncEngine):
    barrier = asyncio.Barrier(2)

    class RacingDocumentService(DocumentService):
        def __init__(self, session: AsyncSession):
            super().__init__(session)
            self.first_lookup = True

        async def get_by_source_url(
            self, graph_id: str, source_url: str
        ) -> Document | None:
            found = await super().get_by_source_url(graph_id, source_url)
            if self.first_lookup:
                self.first_lookup = False
                await barrier.wait()
            return found

    engine = create_async_engine(test_engine.url, pool_size=2, max_overflow=0)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as first_session, sessions() as second_session:
            first, second = await asyncio.wait_for(
                asyncio.gather(
                    RacingDocumentService(first_session).create_or_update_for_source(
                        graph_id="graph",
                        title="first",
                        source_url="https://example.com/shared",
                    ),
                    RacingDocumentService(second_session).create_or_update_for_source(
                        graph_id="graph",
                        title="second",
                        source_url="https://example.com/shared",
                    ),
                ),
                timeout=10,
            )
            assert first.id == second.id
            count = await first_session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.source_url == "https://example.com/shared")
            )
            assert count == 1
    finally:
        await engine.dispose()
