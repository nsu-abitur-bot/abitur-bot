from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from db.postgres.models import Faculty, Program


def _run_migration(connection: Connection, direction: str) -> None:
    path = Path(__file__).resolve().parents[2] / "db/alembic"
    directory = ScriptDirectory(str(path))
    revision = directory.get_revision("0af365dca483")
    assert revision is not None
    with Operations.context(MigrationContext.configure(connection)):
        getattr(revision.module, direction)()


async def test_seat_count_migration_preserves_historical_records(
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
                    id, program_id, year, admission_basis, passing_score, average_score,
                    source_url, created_at, updated_at
                ) VALUES (
                    'legacy', :program, 2024, 'budget', 4.43, 4.51,
                    'https://example.test/2024', '2024-09-01', '2024-09-02'
                )
                """
            ),
            {"program": program.id},
        )
        fetch_legacy = text("SELECT * FROM admission_score WHERE id = 'legacy'")
        legacy = (await connection.execute(fetch_legacy)).mappings().one()
        await connection.run_sync(_run_migration, "upgrade")
        assert (await connection.execute(fetch_legacy)).mappings().one() == {
            **legacy,
            "seat_count": None,
        }
        await connection.run_sync(_run_migration, "downgrade")
        assert (await connection.execute(fetch_legacy)).mappings().one() == legacy
        await connection.run_sync(_run_migration, "upgrade")

        update_seats = text("UPDATE admission_score SET seat_count = :value")
        with pytest.raises(IntegrityError, match="ck_admission_score_seat_count"):
            async with connection.begin_nested():
                await connection.execute(update_seats, {"value": -1})
        await connection.execute(update_seats, {"value": 0})
        with pytest.raises(DBAPIError, match="Cannot downgrade admission records"):
            async with connection.begin_nested():
                await connection.run_sync(_run_migration, "downgrade")
        assert (await connection.execute(fetch_legacy)).mappings().one() == {
            **legacy,
            "seat_count": 0,
        }
