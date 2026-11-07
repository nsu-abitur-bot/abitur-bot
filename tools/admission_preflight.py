"""Check model listing and nine source HEAD requests; no generation or embeddings."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
sys.path.insert(0, "/app")
logging.disable(logging.CRITICAL)

import httpx  # noqa: E402

from bootstrap.runtime_settings import refresh_settings  # noqa: E402
from db.postgres.db import engine  # noqa: E402
from llm.providers.deepseek import DeepSeekProvider  # noqa: E402
from parser.safe_fetch import fetch_url_async  # noqa: E402


async def main() -> None:
    await refresh_settings()
    result: dict[str, Any] = {"sources": [], "generation_calls": 0, "embedding_calls": 0}
    provider = DeepSeekProvider()
    try:
        async with asyncio.timeout(20):
            models = await provider.client.models.list()
        result["deepseek_flash_listed"] = any(
            m.id == "deepseek-flash" for m in models.data
        )
    except Exception as exc:
        result["models_error_type"] = type(exc).__name__
    finally:
        await provider.aclose()
    registry = json.loads(
        Path("/evaluation/dataset/nsu_admission_2026_registry.json").read_text()
    )
    for source_id, source in registry["sources"].items():
        row: dict[str, Any] = {"id": source_id, "url": source["url"]}
        try:
            response = await fetch_url_async(source["url"], method="HEAD", timeout=8)
            row["status"] = response.status_code
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                row["status"] = exc.response.status_code
        result["sources"].append(row)
        print(json.dumps(row), flush=True)
    Path("/reports/preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}))


if __name__ == "__main__":

    async def run() -> None:
        try:
            await main()
        finally:
            await engine.dispose()

    asyncio.run(run())
