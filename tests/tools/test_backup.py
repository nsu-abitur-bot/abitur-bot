"""Резервное копирование БД и базы знаний (#229).

Слепок = pg_dump + архив LightRAG + manifest, собирается во временном
каталоге и получает финальное имя только после успеха. Проверяем ротацию,
полноту слепка, очистку при сбое и восстановление БЗ из архива.
"""

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tools import backup, restore

PG_DUMP_AVAILABLE = shutil.which("pg_dump") is not None


def _make_kb(base: Path) -> dict[str, bytes]:
    files = {
        "abitur_kb/graph_chunk_entity_relation.graphml": b"<graph/>",
        "abitur_kb/vdb_chunks.json": b'{"chunks": 124}',
        "abitur_kb/kv_store_full_docs.json": b'{"doc": "..."}',
    }
    for rel, payload in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return files


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_prune_keeps_last_n_and_ignores_foreign(tmp_path: Path):
    for name in (
        "abitur-20260101-000000",
        "abitur-20260102-000000",
        "abitur-20260103-000000",
        "unrelated",
    ):
        (tmp_path / name).mkdir()

    removed = backup.prune_backups(tmp_path, keep=2)

    assert sorted(p.name for p in removed) == ["abitur-20260101-000000"]
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [
        "abitur-20260102-000000",
        "abitur-20260103-000000",
        "unrelated",
    ]


def test_backup_once_creates_complete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path / "data" / "lightrag"))
    _make_kb(tmp_path / "data" / "lightrag")

    def fake_pg_dump(conn, dump_path: Path) -> None:
        dump_path.write_bytes(b"PGDMP-fake")

    monkeypatch.setattr(backup, "run_pg_dump", fake_pg_dump)
    backups = tmp_path / "backups"

    final = backup.backup_once(backups, keep=5)

    assert final.name.startswith("abitur-")
    assert (final / "postgres.dump").read_bytes() == b"PGDMP-fake"
    assert (final / "lightrag.tar.gz").is_file()
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["postgres_dump"] == "postgres.dump"
    assert manifest["lightrag_archive"] == "lightrag.tar.gz"
    assert manifest["lightrag_files"] == 3
    assert manifest["postgres_bytes"] == len(b"PGDMP-fake")
    # Временных каталогов не осталось.
    assert not list(backups.glob(".tmp-*"))


def test_backup_once_without_kb_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path / "no-such-kb"))

    def fake_pg_dump(conn, dump_path: Path) -> None:
        dump_path.write_bytes(b"PGDMP-fake")

    monkeypatch.setattr(backup, "run_pg_dump", fake_pg_dump)

    final = backup.backup_once(tmp_path / "backups", keep=5)

    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lightrag_archive"] is None
    assert manifest["lightrag_files"] is None
    assert (final / "lightrag.tar.gz").exists() is False


def test_backup_once_dump_failure_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_pg_dump(conn, dump_path: Path) -> None:
        raise RuntimeError("pg_dump failed")

    monkeypatch.setattr(backup, "run_pg_dump", failing_pg_dump)
    backups = tmp_path / "backups"

    with pytest.raises(RuntimeError, match="pg_dump failed"):
        backup.backup_once(backups, keep=5)

    assert not list(backups.iterdir())


def test_archive_and_restore_lightrag_roundtrip(tmp_path: Path):
    base = tmp_path / "data" / "lightrag"
    original = _make_kb(base)

    archive = tmp_path / "kb.tar.gz"
    assert backup.archive_lightrag(base, archive) == len(original)

    # Восстановление в чистый родительский каталог (переезд на новый сервер).
    target_parent = tmp_path / "new-host" / "data"
    target_parent.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("lightrag/")]
        tar.extractall(target_parent, members=members, filter="data")

    assert _hashes(target_parent / "lightrag") == _hashes(base)


def test_restore_lightrag_moves_current_aside(tmp_path: Path):
    base = tmp_path / "data" / "lightrag"
    _make_kb(base)
    archive = tmp_path / "kb.tar.gz"
    backup.archive_lightrag(base, archive)

    aside = restore.restore_lightrag(archive, base)

    assert aside.is_dir()
    assert aside.name.startswith("lightrag.pre-restore-")
    assert _hashes(base) == _hashes(aside)


def test_pg_dump_command_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://abit:secret-pass@pg-host:5433/abitur",
    )
    calls: dict = {}

    def fake_run(argv, env, **kwargs):
        calls["argv"] = argv
        calls["env"] = env
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(backup.subprocess, "run", fake_run)

    backup.run_pg_dump(backup.pg_connection(), tmp_path / "dump")

    argv = calls["argv"]
    assert argv[0] == "pg_dump"
    for flag, value in (
        ("-h", "pg-host"),
        ("-p", "5433"),
        ("-U", "abit"),
        ("-d", "abitur"),
        ("-f", str(tmp_path / "dump")),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "-Fc" in argv
    # Пароль уходит через PGPASSWORD, а не через argv (виден в ps).
    assert "secret-pass" not in argv
    assert calls["env"]["PGPASSWORD"] == "secret-pass"


@pytest.mark.skipif(not PG_DUMP_AVAILABLE, reason="в окружении нет pg_dump")
async def test_dump_restore_roundtrip_real_postgres(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Регрессия #229 на реальном PostgreSQL: дамп должен восстанавливаться
    поверх базы (режим --clean) без остаточных артефактов."""
    from tests.postgres.conftest import TEST_DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS backup_probe_229"))
        await conn.execute(
            text("CREATE TABLE backup_probe_229 (payload text PRIMARY KEY)")
        )
        await conn.execute(text("INSERT INTO backup_probe_229 VALUES ('must-survive')"))
    await engine.dispose()

    conn_info = backup.pg_connection()
    assert conn_info["dbname"]  # подключение разобралось из TEST_DATABASE_URL

    dump_path = tmp_path / "postgres.dump"
    backup.run_pg_dump(conn_info, dump_path)
    assert dump_path.stat().st_size > 0

    # Портим базу так же, как реальный сбой: таблица исчезла.
    engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE backup_probe_229"))
    await engine.dispose()

    restore.restore_postgres(dump_path, conn_info)

    engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    async with async_sessionmaker(engine)() as session:
        value = await session.execute(text("SELECT payload FROM backup_probe_229"))
        assert value.scalar_one() == "must-survive"
        await session.execute(text("DROP TABLE backup_probe_229"))
    await engine.dispose()
