"""Восстановление PostgreSQL и базы знаний из слепка tools/backup (#229).

Сценарии — docs/backup.md: откат на прошлое состояние и переезд на другой
сервер. Перед запуском остановите api/bot/scheduler (`docker compose stop`),
иначе активные соединения не дадут пересоздать объекты базы, а бот начнёт
писать в восстанавливаемую базу знаний.
"""

import argparse
import json
import logging
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from config.instance import get_setting
from tools.backup import pg_connection, pg_env

logger = logging.getLogger(__name__)


def restore_postgres(dump_path: Path, conn: dict) -> None:
    """Разворачивает дамп поверх базы: --clean удаляет существующие объекты,
    --if-exists не даёт упасть на отсутствующих."""
    argv = [
        "pg_restore",
        "--exit-on-error",
        "--no-owner",
        "--clean",
        "--if-exists",
        "-h",
        conn["host"],
        "-p",
        str(conn["port"]),
        "-U",
        conn["user"],
        "-d",
        conn["dbname"],
        str(dump_path),
    ]
    completed = subprocess.run(
        argv, env=pg_env(conn["password"]), capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pg_restore завершился с кодом {completed.returncode}: {completed.stderr}"
        )


def restore_lightrag(archive_path: Path, kb_base: Path) -> Path:
    """Распаковывает архив БЗ, текущий каталог не удаляет, а отодвигает.

    Отодвинутый каталог — страховка: если восстановление оказалось ошибкой,
    прежнее состояние всё ещё на месте.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aside_dir = kb_base.with_name(f"{kb_base.name}.pre-restore-{stamp}")
    if kb_base.is_dir():
        kb_base.rename(aside_dir)
    kb_base.parent.mkdir(parents=True, exist_ok=True)
    # Архив содержит корень lightrag/ относительно ./data — распаковываем
    # в родителя базы, filter отбирает только обычные файлы и каталоги.
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(kb_base.parent, filter="data")
    return aside_dir


def _load_manifest(backup_dir: Path) -> dict:
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"В {backup_dir} нет manifest.json — это не слепок?")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_dir", type=Path, help="каталог слепка из ./backups")
    parser.add_argument(
        "--yes", action="store_true", help="не спрашивать подтверждение (для скриптов)"
    )
    parser.add_argument(
        "--no-db", action="store_true", help="не трогать PostgreSQL (только база знаний)"
    )
    parser.add_argument(
        "--no-kb", action="store_true", help="не трогать базу знаний (только PostgreSQL)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = _load_manifest(args.backup_dir)
    conn = pg_connection()
    kb_base = Path(get_setting("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
    archive_name = manifest.get("lightrag_archive")

    print(f"Слепок от {manifest.get('created_at', '?')}:")
    print(f"  PostgreSQL → {conn['dbname']}@{conn['host']}:{conn['port']}")
    print(f"  База знаний → {kb_base}" + (" (нет в слепке)" if not archive_name else ""))
    if not args.yes and input("Восстановить? Введите `да`: ").strip().lower() != "да":
        print("Отменено.")
        return 1

    if not args.no_db:
        restore_postgres(args.backup_dir / manifest["postgres_dump"], conn)
        logger.info("PostgreSQL восстановлен")

    if not args.no_kb and archive_name:
        aside = restore_lightrag(args.backup_dir / archive_name, kb_base)
        logger.info("База знаний восстановлена; прежняя отодвинута в %s", aside)

    print("Готово. Запустите сервисы: docker compose up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
