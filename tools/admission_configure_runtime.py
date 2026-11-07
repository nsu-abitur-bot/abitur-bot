"""Initialize the empty evaluation instance from its administrator-provided env."""

import asyncio
import json
import logging
import os
import sys

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
sys.path.insert(0, "/app")
logging.disable(logging.CRITICAL)

from sqlalchemy import text  # noqa: E402

from db.postgres.db import AsyncSessionLocal, engine  # noqa: E402
from db.postgres.services.instance_settings import InstanceSettingsService  # noqa: E402


async def main() -> None:
    """Preserve settings, fill only the two approved providers, emit no credentials."""
    async with AsyncSessionLocal() as session:
        assert engine.url.host == "postgres" and engine.url.database == "eval388"
        for table in ("document", "content_proposal", "faq_entry", "user", "message"):
            count = await session.scalar(text(f'SELECT count(*) FROM "{table}"'))
            assert count == 0, "Evaluation instance is not empty"
        service = InstanceSettingsService(session)
        snapshot = await service.read()
        assert snapshot.revision == 1, "Configuration already changed: inspect again"
        values = dict(snapshot.values)
        assert not values.get("DEEPSEEK_API_KEY") and not values.get("GIGACHAT_API_KEY")
        # Consume the injected application configuration in process; no files,
        # credentials, decrypted snapshots or upstream bodies are emitted.
        for name in ("DEEPSEEK_API_KEY", "GIGACHAT_API_KEY"):
            assert os.environ.get(name), "Required provider is absent from runtime"
            values[name] = os.environ[name]
        for name in ("GIGACHAT_SCOPE", "DEEPSEEK_BASE_URL"):
            if os.environ.get(name):
                values[name] = os.environ[name]
        values.update(
            LLM_PROVIDER="deepseek",
            LIGHTRAG_LLM_PROVIDER="deepseek",
            LIGHTRAG_EMBEDDING_PROVIDER="gigachat",
            FAQ_EMBEDDING_PROVIDER="gigachat",
            DEEPSEEK_MODEL="deepseek-flash",
            DEEPSEEK_GRAPH_MODEL="deepseek-flash",
            GIGACHAT_EMBEDDING_MODEL="EmbeddingsGigaR",
            TELEGRAM_ENABLED="false",
            MAX_ENABLED="false",
        )
        updated = await service.write(values, snapshot.revision)
        print(json.dumps({"configured": True, "revision": updated.revision}))


if __name__ == "__main__":

    async def run() -> None:
        try:
            await main()
        finally:
            await engine.dispose()

    try:
        asyncio.run(run())
    except Exception as exc:
        print(json.dumps({"configured": False, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
