"""Exercise the uniqueness migration against populated PostgreSQL tables."""

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from db.postgres.models import Document


def run_migration(connection: Connection, direction: str) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "db/alembic/versions/2026_11_04_0100-document_source_url_unique.py"
    )
    spec = importlib.util.spec_from_file_location("source_url_migration", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, direction)()


@pytest.mark.asyncio
async def test_upgrade_reconciles_duplicates_without_losing_metadata(
    session: AsyncSession, test_engine: AsyncEngine
):
    await session.execute(text("DROP INDEX ix_document_graph_source_url_active"))
    rows = [
        Document(
            id=document_id,
            title=document_id,
            graph_id=graph,
            rag_doc_id=f"rag-{document_id}",
            source_url=url,
            status=status,
            content_hash=f"hash-{document_id}",
            source_proposal_id=f"proposal-{document_id}",
            updated_at=updated_at,
        )
        for document_id, graph, url, status, updated_at in (
            ("old", "g", "https://example.com", "indexed", datetime(2026, 1, 1)),
            ("tie-a", "g", "https://example.com", "changed", datetime(2026, 1, 2)),
            ("tie-b", "g", "https://example.com", "indexed", datetime(2026, 1, 2)),
            ("deleted-en", "g", "https://example.com", "deleted", datetime(2026, 1, 3)),
            ("deleted-ru", "g", "https://example.com", "удалён", datetime(2026, 1, 3)),
            (
                "other-graph",
                "other",
                "https://example.com",
                "indexed",
                datetime(2026, 1, 1),
            ),
            ("other-url", "g", "https://example.org", "pending", datetime(2026, 1, 1)),
            ("empty-a", "g", "", "нет ссылки", datetime(2026, 1, 1)),
            ("empty-b", "g", "", "нет ссылки", datetime(2026, 1, 1)),
            ("null-a", "g", None, "нет ссылки", datetime(2026, 1, 1)),
            ("null-b", "g", None, "нет ссылки", datetime(2026, 1, 1)),
        )
    ]
    session.add_all(rows)
    await session.commit()
    original = {
        row.id: {
            column.name: getattr(row, column.name)
            for column in Document.__table__.columns
        }
        for row in rows
    }

    async with test_engine.begin() as connection:
        await connection.run_sync(run_migration, "upgrade")

    session.expire_all()
    migrated = (await session.scalars(select(Document))).all()
    assert len(migrated) == len(original)
    for row in migrated:
        expected = original[row.id].copy()
        if row.id in {"old", "tie-a"}:
            expected["status"] = "удалён"
        assert {
            column.name: getattr(row, column.name)
            for column in Document.__table__.columns
        } == expected
    await session.rollback()

    # The migrated index must enforce uniqueness while allowing cleared URLs.
    async with test_engine.begin() as connection:
        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(
                    insert(Document).values(
                        title="duplicate",
                        graph_id="g",
                        source_url="https://example.com",
                        rag_doc_id="new-rag-id",
                    )
                )
        for document_id in ("tie-b", "other-url"):
            await connection.execute(
                text("UPDATE document SET source_url = '' WHERE id = :id"),
                {"id": document_id},
            )
        await connection.run_sync(run_migration, "downgrade")
        assert (
            await connection.scalar(
                text("SELECT to_regclass('ix_document_graph_source_url_active')")
            )
            is None
        )
        await connection.run_sync(run_migration, "upgrade")
