"""Offline Cloud.ru migration; keys enter via stdin, never argv or logs."""

import argparse
import asyncio
import json
import sys

from tools.reembed_gigachat import migrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not args.offline:
        parser.error("Сначала остановите API, bot и scheduler; требуется --offline")
    changes = json.load(sys.stdin)
    if (
        not isinstance(changes, dict)
        or set(changes) - {"CLOUDRU_API_KEY", "CLOUDRU_EMBEDDING_MODEL"}
        or any(not isinstance(v, str) for v in changes.values())
    ):
        parser.error("Ожидаются только настройки подключения Cloud.ru")
    try:
        print(
            json.dumps(
                asyncio.run(migrate(changes, provider="cloudru")), ensure_ascii=False
            )
        )
    except Exception as exc:
        print(
            "Пересчёт не завершён: "
            + type(exc).__name__
            + ". Экземпляр оставьте остановленным; "
            + "проверьте каталог embedding-migration-*.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
