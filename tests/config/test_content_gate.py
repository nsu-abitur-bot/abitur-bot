import json

import pytest

from config.content_gate import ModelMigrationBusy, content_write, migration_marker


@pytest.mark.asyncio
async def test_writes_are_rejected_during_migration_and_resume_afterwards(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path))
    async with content_write():
        pass
    migration_marker().write_text(json.dumps({"id": "test-job"}))
    with pytest.raises(ModelMigrationBusy, match="пересчёт"):
        async with content_write():
            pytest.fail("writer entered while a migration is active")
    migration_marker().unlink()
    async with content_write():
        pass
