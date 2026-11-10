"""Резервное копирование PostgreSQL и файловой базы знаний (#229).

Запускается отдельным compose-сервисом `backup`: периодически снимает
согласованный дамп базы (`pg_dump -Fc`) и архив каталога LightRAG
(graphml, vdb/kv json). Копии лежат в ./backups по каталогу на слепок,
ротация — по числу последних (BACKUP_KEEP). Восстановление —
`python -m tools.restore <слепок>`, процедура — docs/backup.md.

Каталог временной копии называется с префиксом точки и переименовывается
в финальное имя только после успеха: оборванный бэкап нельзя перепутать
с полным.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from config.instance import get_setting
from db.postgres.config import get_database_url

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = "./backups"
DEFAULT_BACKUP_KEEP = 14
DEFAULT_INTERVAL_SECONDS = 86400

BACKUP_PREFIX = "abitur-"
TMP_PREFIX = ".tmp-"


def pg_connection() -> dict:
    """Параметры подключения для pg_dump/pg_restore из настроек приложения.

    URL из db.postgres.config покрывает оба способа задания (DATABASE_URL
    целиком или отдельные DB_*): pg_dump нужны host/port/user/dbname и
    пароль в PGPASSWORD, а не URL.
    """
    url = get_database_url().replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlsplit(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": unquote(parsed.password or ""),
        "dbname": parsed.path.lstrip("/"),
    }


def pg_env(password: str) -> dict:
    # Пароль через окружение: в argv он попал бы в вывод ps и в логи compose.
    return {**os.environ, "PGPASSWORD": password}


def run_pg_dump(conn: dict, dump_path: Path) -> None:
    """Согласованный дамп базы в custom-формате: он сжимается и подходит
    для pg_restore с выборочным восстановлением."""
    argv = [
        "pg_dump",
        "-h",
        conn["host"],
        "-p",
        str(conn["port"]),
        "-U",
        conn["user"],
        "-Fc",
        "-f",
        str(dump_path),
        "-d",
        conn["dbname"],
    ]
    completed = subprocess.run(
        argv, env=pg_env(conn["password"]), capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pg_dump завершился с кодом {completed.returncode}: {completed.stderr}"
        )


def archive_lightrag(base_dir: Path, archive_path: Path) -> int:
    """Архивирует файловую базу знаний (граф + векторные/kv-хранилища).

    Возвращает число заархивированных файлов. Слепок снимается на живом
    каталоге: редкая запись в момент чтения файла может попасть в архив
    неполной, поэтому восстановление проверяется по manifest.json
    (число файлов) и при необходимости tools/release_restore_probe.py.
    """
    file_count = 0
    with tarfile.open(archive_path, "x:gz") as tar:
        for path in sorted(base_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(base_dir.parent))
                file_count += 1
    return file_count


def write_manifest(
    manifest_path: Path,
    *,
    created_at: datetime,
    kb_files: int | None,
    dump_path: Path,
    archive_path: Path | None,
) -> None:
    manifest = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "postgres_dump": dump_path.name,
        "postgres_bytes": dump_path.stat().st_size,
        "lightrag_archive": archive_path.name if archive_path else None,
        "lightrag_files": kb_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prune_backups(backup_root: Path, keep: int) -> list[Path]:
    """Оставляет последние `keep` полных копий, старые удаляет.

    Имена содержат метку времени UTC, поэтому сортировка по имени — это
    сортировка по времени. Чужие каталоги (без префикса abitur-) не трогаем.
    """
    if keep <= 0:
        return []
    backups = sorted(
        p
        for p in backup_root.iterdir()
        if p.is_dir() and p.name.startswith(BACKUP_PREFIX)
    )
    removed = []
    for old in backups[:-keep]:
        shutil.rmtree(old)
        removed.append(old)
    return removed


def backup_once(backup_root: Path, keep: int = DEFAULT_BACKUP_KEEP) -> Path:
    """Один слепок: дамп БД + архив БЗ + manifest, затем ротация.

    Поднимает исключение при сбое pg_dump — копия без базы не имеет смысла.
    """
    conn = pg_connection()
    kb_base = Path(get_setting("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))

    created_at = datetime.now(timezone.utc)
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=True)
    # Обрывы прошлых запусков: одиночный сервис не пишет параллельно,
    # забытые временные каталоги можно убирать смело.
    for stale in backup_root.glob(f"{TMP_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)
    tmp_dir = backup_root / f"{TMP_PREFIX}{stamp}"
    tmp_dir.mkdir(parents=True)

    dump_path = tmp_dir / "postgres.dump"
    try:
        run_pg_dump(conn, dump_path)

        archive_path = None
        kb_files = None
        if kb_base.is_dir():
            archive_path = tmp_dir / "lightrag.tar.gz"
            kb_files = archive_lightrag(kb_base, archive_path)
        else:
            logger.warning(
                "Каталог базы знаний %s не найден: слепок содержит только дамп БД",
                kb_base,
            )

        write_manifest(
            tmp_dir / "manifest.json",
            created_at=created_at,
            kb_files=kb_files,
            dump_path=dump_path,
            archive_path=archive_path,
        )
        final_dir = backup_root / f"{BACKUP_PREFIX}{stamp}"
        tmp_dir.rename(final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    removed = prune_backups(backup_root, keep)
    logger.info(
        "Слепок %s готов (БД: %d Б, БЗ: %s файлов); ротация удалила %d старых",
        final_dir,
        (final_dir / "postgres.dump").stat().st_size,
        kb_files if kb_files is not None else "—",
        len(removed),
    )
    return final_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="снять один слепок и выйти (ручной запуск), без цикла ожидания",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backup_root = Path(get_setting("BACKUP_DIR", DEFAULT_BACKUP_DIR))
    keep = int(get_setting("BACKUP_KEEP", str(DEFAULT_BACKUP_KEEP)))
    interval = int(get_setting("BACKUP_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)))

    while True:
        try:
            backup_once(backup_root, keep=keep)
        except Exception:
            logger.exception("Слепок не удался; следующий через %d c", interval)
        if args.once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
