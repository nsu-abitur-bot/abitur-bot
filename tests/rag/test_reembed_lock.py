from pathlib import Path

import pytest

from tools import reembed_gigachat


async def test_parallel_migration_is_rejected_before_reading_or_publishing(
    monkeypatch, tmp_path
):
    root = tmp_path / "index"
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(root))
    called = []

    async def forbidden(changes, target):
        called.append(True)
        return {}

    monkeypatch.setattr(reembed_gigachat, "_migrate", forbidden)
    with reembed_gigachat._migration_lock(root):
        with pytest.raises(BlockingIOError):
            await reembed_gigachat.migrate({})
    assert not called
    await reembed_gigachat.migrate({})
    assert called == [True]


async def test_lock_is_held_until_migration_finishes(monkeypatch, tmp_path):
    root = tmp_path / "index"
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(root))

    async def migrate(changes, target: Path):
        with pytest.raises(BlockingIOError):
            with reembed_gigachat._migration_lock(target):
                pytest.fail("parallel lock acquired")
        return {"done": True}

    monkeypatch.setattr(reembed_gigachat, "_migrate", migrate)
    assert await reembed_gigachat.migrate({}) == {"done": True}
    with reembed_gigachat._migration_lock(root):
        pass
