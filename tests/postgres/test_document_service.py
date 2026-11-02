import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
