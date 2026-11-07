"""Read-only evidence from an isolated #388 container, without secret output."""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
sys.path.insert(0, "/app")
logging.disable(logging.CRITICAL)

from sqlalchemy import text  # noqa: E402

from bootstrap.runtime_settings import refresh_settings  # noqa: E402
from config import instance  # noqa: E402
from config.instance import get_setting  # noqa: E402
from db.postgres.db import engine  # noqa: E402
from db.redis.client import RedisClient  # noqa: E402


async def inspect_runtime(variant: str) -> dict:
    """Return counts, mount identities and an explicit allowlist of settings."""
    await refresh_settings()
    allowed_settings = (
        "LLM_PROVIDER",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_GRAPH_MODEL",
        "LIGHTRAG_LLM_PROVIDER",
        "LIGHTRAG_LLM_MODEL",
        "LIGHTRAG_EMBEDDING_PROVIDER",
        "FAQ_EMBEDDING_PROVIDER",
        "GIGACHAT_EMBEDDING_MODEL",
        "PDF_PARSER_PROVIDER",
        "TELEGRAM_ENABLED",
        "MAX_ENABLED",
        "LIGHTRAG_STORAGE",
        "LIGHTRAG_WORKSPACE_BASE",
        "GIGACHAT_LOCK_DIRECTORY",
    )
    settings = {name: get_setting(name) for name in allowed_settings}
    credential_configured = {
        name: bool(get_setting(name)) for name in ("DEEPSEEK_API_KEY", "GIGACHAT_API_KEY")
    }
    tables = (
        "document",
        "content_proposal",
        "faq_entry",
        "user",
        "message",
        "message_logs",
        "admins",
    )
    async with engine.connect() as conn:
        counts = {
            table: (
                await conn.execute(text(f'SELECT count(*) FROM "{table}"'))
            ).scalar_one()
            for table in tables
        }
        server = (
            await conn.execute(
                text("SELECT inet_server_addr()::text, current_database()")
            )
        ).one()
    targets = {
        "/app/data",
        "/app/logs",
        "/app/branding-media",
        "/app/provider-locks",
        "/evaluation",
        "/reports",
    }
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if fields[4] in targets:
            mounts.append({"root": fields[3], "target": fields[4], "options": fields[5]})
    data_files = []
    root = Path("/app/data")
    if root.exists():
        for file in root.rglob("*"):
            if file.is_file():
                data_files.append(
                    {"path": str(file.relative_to(root)), "bytes": file.stat().st_size}
                )
    cgroup = Path("/sys/fs/cgroup")
    resources = {
        name: (cgroup / name).read_text().strip()
        for name in ("memory.max", "memory.current", "memory.swap.max", "cpu.max")
        if (cgroup / name).is_file()
    }
    cache = RedisClient()
    try:
        redis_keys = await cache.client.dbsize()
        redis_run_id = (await cache.client.info("server"))["run_id"]
    finally:
        await cache.client.aclose()
    return {
        "variant": variant,
        "container_hostname": socket.gethostname(),
        "database_host": engine.url.host,
        "database_server": server[0],
        "database_name": server[1],
        "settings": settings,
        "settings_revision": instance._revision,
        "credential_configured": credential_configured,
        "environment_nonsecret_settings": {
            name: os.getenv(name) for name in allowed_settings
        },
        "row_counts": counts,
        "mounts": mounts,
        "data_files": data_files,
        "redis_keys": redis_keys,
        "redis_run_id": redis_run_id,
        "resources": resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("with-kb", "empty-kb"))
    parser.add_argument("--phase", choices=("before", "after"), default="before")
    args = parser.parse_args()

    async def run() -> dict:
        try:
            return await inspect_runtime(args.variant)
        finally:
            await engine.dispose()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}))
        return 1
    output = Path("/reports") / f"runtime-{args.phase}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
