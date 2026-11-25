"""Сохранение итогов приёма при смене ключа записей в PostgreSQL."""

import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from db.postgres.models import Faculty, Program


def _run_migration(connection: Connection, direction: str) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "db/alembic/versions/2026_11_24_2200-0af365dca482_admission_score_study_form.py"
    )
    spec = importlib.util.spec_from_file_location("admission_score_migration", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, direction)()


async def test_migration_preserves_legacy_scores_and_null_uniqueness(
    session: AsyncSession, test_engine: AsyncEngine
) -> None:
    faculty = Faculty(name="Test faculty")
    session.add(faculty)
    await session.flush()
    program = Program(faculty_id=faculty.id, name="Test program", level="spo")
    session.add(program)
    await session.commit()

    async with test_engine.begin() as connection:
        await connection.run_sync(_run_migration, "downgrade")
        await connection.execute(
            text(
                """
                INSERT INTO admission_score (
                    id, program_id, year, form, passing_score, average_score,
                    source_url, created_at, updated_at
                ) VALUES
                    ('legacy-budget', :program, 2024, 'budget', 4.43, 4.51,
                     'https://example.test/2024', '2024-09-01', '2024-09-02'),
                    ('legacy-paid', :program, 2024, 'paid', 3.89, 4.02,
                     'https://example.test/2024', '2024-09-01', '2024-09-02')
                """
            ),
            {"program": program.id},
        )
        read_scores = text("SELECT * FROM admission_score ORDER BY id")
        legacy = [dict(row) for row in (await connection.execute(read_scores)).mappings()]
        await connection.run_sync(_run_migration, "upgrade")
        expected = [row.copy() for row in legacy]
        for row in expected:
            row["admission_basis"] = row.pop("form")
            row["study_form"] = None
        actual = [dict(row) for row in (await connection.execute(read_scores)).mappings()]
        assert actual == expected

        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(
                    text(
                        """
                        INSERT INTO admission_score (
                            id, program_id, year, admission_basis, created_at, updated_at
                        ) VALUES ('duplicate', :program, 2024, 'budget', now(), now())
                        """
                    ),
                    {"program": program.id},
                )
        await connection.run_sync(_run_migration, "downgrade")
        assert (await connection.execute(read_scores)).mappings().all() == legacy
        await connection.run_sync(_run_migration, "upgrade")
        await connection.execute(
            text("UPDATE admission_score SET study_form = 'full_time' WHERE id = :id"),
            {"id": "legacy-paid"},
        )
        with pytest.raises(DBAPIError, match="Cannot downgrade classified"):
            async with connection.begin_nested():
                await connection.run_sync(_run_migration, "downgrade")
        assert (
            await connection.scalar(
                text("SELECT study_form FROM admission_score WHERE id = 'legacy-paid'")
            )
            == "full_time"
        )
