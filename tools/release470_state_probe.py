"""Check JWT expiry, storage isolation and FAQ notifications in isolated #470.

No provider, bot or production lifespan is started. Only synthetic data is used.
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch


async def probe() -> dict[str, Any]:
    """Use the real app and storage services, retaining only booleans and timings."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["DB_ECHO"] = "false"
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, "/app")

    import httpx
    from sqlalchemy import delete, func, select, text

    from api.auth.security import create_access_token, hash_password
    from api.main import app
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.models import Admin, FaqEntry, Message, User
    from db.postgres.services.faq import FAQ_CHANGE_CHANNEL
    from db.postgres.services.message import MessageService
    from db.redis.client import RedisClient

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected SQL target")
    engine.echo = False
    redis = RedisClient()
    if redis.client.connection_pool.connection_kwargs.get("host") != "redis":
        raise RuntimeError("Unexpected Redis target")
    prefix = "r470_" + secrets.token_hex(8)
    sessions = [prefix + suffix for suffix in (":telegram:a", ":telegram:b")]
    user_ids: list[int] = []
    admin_id = None
    question = prefix + " synthetic FAQ"
    event = asyncio.Event()
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "ASGI without lifespan; real isolated SQL/Redis; synthetic data",
        "checks": checks,
        "provider_calls": 0,
        "messages_sent": 0,
    }

    def record(name: str, passed: bool) -> None:
        checks.append({"name": name, "passed": bool(passed)})

    def notify(*args: Any) -> None:
        event.set()

    try:
        async with AsyncSessionLocal() as db:
            for table in ('"user"', "message"):
                if await db.scalar(text(f"select count(*) from {table}")) != 0:
                    raise RuntimeError("Expected empty synthetic user target")
            result["baseline_admin_count"] = await db.scalar(
                select(func.count()).select_from(Admin)
            )
            admin = Admin(
                username=prefix,
                password_hash=hash_password(secrets.token_urlsafe(24)),
                role="admin",
                is_active=True,
            )
            db.add(admin)
            await db.flush()
            admin_id = admin.id
            for _ in sessions:
                user = User(telegram_id=-(secrets.randbelow(10**12) + 1))
                db.add(user)
                await db.flush()
                user_ids.append(user.user_id)
            await db.commit()

        valid = create_access_token(admin_id)
        # Only this process's token lifetime is patched during issuance. The real
        # decoder and system clock are unchanged when the expired JWT is checked.
        with patch("api.auth.security.TOKEN_EXPIRE_DAYS", -1):
            expired = create_access_token(admin_id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://isolated.invalid"
        ) as client:
            ok = await client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {valid}"}
            )
            bad = await client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}
            )
            record("valid_jwt_accepted", ok.status_code == 200)
            record("expired_signed_jwt_rejected", bad.status_code == 401)

            async with engine.connect() as listener:
                raw = await listener.get_raw_connection()
                driver = raw.driver_connection
                if driver is None:
                    raise RuntimeError("Missing isolated SQL listener connection")
                await driver.add_listener(FAQ_CHANGE_CHANNEL, notify)
                try:
                    headers = {"Authorization": f"Bearer {valid}"}
                    event.clear()
                    start = time.monotonic()
                    created = await client.post(
                        "/api/v1/faq",
                        headers=headers,
                        json={"question": question, "answer": "synthetic v1"},
                    )
                    record("faq_create_http", created.status_code == 201)
                    if created.status_code != 201:
                        raise RuntimeError("Synthetic FAQ creation failed")
                    faq_id = created.json()["id"]
                    await asyncio.wait_for(event.wait(), 5)
                    result["create_to_sql_listener_ms"] = round(
                        (time.monotonic() - start) * 1000, 3
                    )
                    record("faq_create_notify", event.is_set())
                    event.clear()
                    start = time.monotonic()
                    updated = await client.put(
                        f"/api/v1/faq/{faq_id}",
                        headers=headers,
                        json={"question": question, "answer": "synthetic v2"},
                    )
                    record("faq_update_http", updated.status_code == 200)
                    await asyncio.wait_for(event.wait(), 5)
                    result["update_to_sql_listener_ms"] = round(
                        (time.monotonic() - start) * 1000, 3
                    )
                    async with AsyncSessionLocal() as db:
                        row = await db.get(FaqEntry, faq_id)
                        record(
                            "faq_updated_in_separate_sql_session",
                            row is not None and row.answer == "synthetic v2",
                        )
                    record("faq_update_notify", event.is_set())
                finally:
                    await driver.remove_listener(FAQ_CHANGE_CHANNEL, notify)

        async with AsyncSessionLocal() as db:
            service = MessageService(db)
            saved = []
            for uid, sid in zip(user_ids, sessions):
                message = await service.create_message(
                    uid, sid, "synthetic question", "synthetic answer"
                )
                if message is None:
                    raise RuntimeError("Synthetic message write failed")
                saved.append(message.id)
            for index, (uid, sid) in enumerate(zip(user_ids, sessions)):
                by_user = await service.get_messages_by_user(uid)
                by_session = await service.get_messages_by_session(sid)
                record(
                    f"sql_user_{index}_isolated",
                    [m.id for m in by_user] == [saved[index]],
                )
                record(
                    f"sql_session_{index}_isolated",
                    [m.id for m in by_session] == [saved[index]],
                )

        for i, sid in enumerate(sessions):
            await redis.add_message(sid, {"role": "user", "content": f"synthetic-{i}"})
        for i, sid in enumerate(sessions):
            record(
                f"redis_session_{i}_isolated",
                await redis.get_history(sid)
                == [{"role": "user", "content": f"synthetic-{i}"}],
            )
        reset = await redis.reset_dialog_session(sessions[0])
        sessions.append(reset)
        record("reset_new_session_empty", await redis.get_history(reset) == [])
        record("reset_old_session_empty", await redis.get_history(sessions[0]) == [])
        record(
            "reset_other_session_unchanged",
            len(await redis.get_history(sessions[1])) == 1,
        )
    finally:
        keys = [f"chat_history:{s}" for s in sessions]
        keys += [f"session_state:{s}:dialog_generation" for s in sessions[:2]]
        await redis.client.delete(*keys)
        result["redis_cleanup_verified"] = await redis.client.exists(*keys) == 0
        await redis.close()
        async with AsyncSessionLocal() as db:
            await db.execute(delete(Message).where(Message.user_id.in_(user_ids)))
            await db.execute(delete(User).where(User.user_id.in_(user_ids)))
            await db.execute(delete(FaqEntry).where(FaqEntry.question == question))
            if admin_id:
                await db.execute(delete(Admin).where(Admin.id == admin_id))
            await db.commit()
            result["sql_cleanup_verified"] = all(
                [
                    await db.scalar(select(func.count()).select_from(User)) == 0,
                    await db.scalar(select(func.count()).select_from(Message)) == 0,
                    await db.scalar(
                        select(func.count())
                        .select_from(FaqEntry)
                        .where(FaqEntry.question == question)
                    )
                    == 0,
                ]
            )
            result["final_admin_count"] = await db.scalar(
                select(func.count()).select_from(Admin)
            )
        await engine.dispose()
    result["passed"] = (
        all(c["passed"] for c in checks)
        and result["sql_cleanup_verified"]
        and result["redis_cleanup_verified"]
    )
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--isolated-primary"] or not Path("/evaluation").is_dir():
        raise SystemExit("Use only the isolated primary runner")
    try:
        output = asyncio.run(asyncio.wait_for(probe(), 150))
        print(json.dumps(output))
        raise SystemExit(0 if output["passed"] else 1)
    except Exception as exc:
        print(json.dumps({"probe_error": type(exc).__name__}))
        raise SystemExit(1) from None
