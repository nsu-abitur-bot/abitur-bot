"""Sequential isolated ASGI FAQ edits in the metered worker.

Creates one temporary test admin, logs in through the real API and removes only
its own FAQ/admin. No credentials, identifiers or response bodies leave this process.
"""

import asyncio
import json
import logging
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path


async def api_call(
    app, method: str, path: str, body: dict | None = None, token: str = ""
):
    data = json.dumps(body).encode() if body is not None else b""
    headers = [(b"content-type", b"application/json")]
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "client": ("127.0.0.1", 470),
        "server": ("eval470", 80),
    }
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": data, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = next(
        item["status"] for item in sent if item["type"] == "http.response.start"
    )
    content = b"".join(
        item.get("body", b"") for item in sent if item["type"] == "http.response.body"
    )
    return status, json.loads(content) if content else {}


async def operate(
    operation: str,
    *,
    manifest_path: Path | None = None,
    refresh_before: bool = True,
) -> dict:
    from sqlalchemy import func, select

    from api.auth.security import hash_password
    from api.main import app
    from bootstrap.runtime_settings import runtime_settings
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.models import Admin, FaqEntry
    from faq.matcher import get_faq_matcher
    from tools.release470_store import BudgetStore

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected API test target")
    plan = json.loads(Path("/reports/pilot-plan-approved-2026-11-21.json").read_text())
    private_manifest = manifest_path or Path("/reports/faq-serial-records-20261121.json")
    if operation == "create":
        descriptor = os.open(
            private_manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        os.close(descriptor)
        manifest = {"faq_id": None, "cleaned": False}
    else:
        manifest = json.loads(private_manifest.read_text())
        if manifest.get("cleaned"):
            raise RuntimeError("This test record has already been cleaned")
    username = "EVAL470-" + secrets.token_hex(8)
    password = secrets.token_urlsafe(24)
    result: dict = {
        "operation": operation,
        "network_listener_tested": False,
        "authentication": "real API login with disposable isolated admin",
    }
    async with runtime_settings():
        async with AsyncSessionLocal() as db:
            baseline = await db.scalar(select(func.count()).select_from(FaqEntry))
            if operation == "create":
                manifest["baseline_faq_count"] = baseline
            else:
                own_faq = await db.get(FaqEntry, manifest["faq_id"])
                if own_faq is None or own_faq.question != plan["questions"]["faq-1"]:
                    raise RuntimeError("The saved record is not the owned test FAQ")
            admin = Admin(
                username=username,
                password_hash=hash_password(password),
                role="admin",
                is_active=True,
            )
            db.add(admin)
            await db.commit()
            await db.refresh(admin)
            admin_id = admin.id
            manifest["temporary_admin_id"] = str(admin_id)
            private_manifest.write_text(json.dumps(manifest))
        try:
            status, response = await api_call(
                app,
                "POST",
                "/api/v1/auth/login",
                {
                    "username": username,
                    "password": password,
                },
            )
            if status != 200:
                raise RuntimeError("Synthetic API login failed")
            token = response["access_token"]
            # Cleanup must remain provider-free even after the lease is revoked.
            if operation != "cleanup" and refresh_before:
                matcher = get_faq_matcher()
                await matcher.refresh_from_db()
            payload = {
                "question": plan["questions"]["faq-1"],
                "aliases": [plan["questions"]["faq-2"]],
                "answer": "EVAL470 synthetic answer v"
                + ("1" if operation == "create" else "2"),
            }
            if operation == "create":
                status, body = await api_call(app, "POST", "/api/v1/faq", payload, token)
                if status != 201:
                    raise RuntimeError("Synthetic FAQ creation failed")
                manifest["faq_id"] = body["id"]
            elif operation == "update":
                matcher = get_faq_matcher()
                status, _ = await api_call(
                    app, "PUT", f"/api/v1/faq/{manifest['faq_id']}", payload, token
                )
                if status != 200:
                    raise RuntimeError("Synthetic FAQ update failed")
                questions, answers, _ = matcher._index
                index = questions.index(payload["question"])
                result["api_cache_matches_new_answer_without_reload"] = (
                    answers[index] == payload["answer"]
                )
            else:
                status, _ = await api_call(
                    app, "DELETE", f"/api/v1/faq/{manifest['faq_id']}", token=token
                )
                if status != 204:
                    raise RuntimeError("Synthetic FAQ cleanup failed")
                manifest["cleaned"] = True
            result["status"] = status
            result["committed_at_utc"] = datetime.now(UTC).isoformat()
            private_manifest.write_text(json.dumps(manifest))
        finally:
            async with AsyncSessionLocal() as db:
                own_admin = await db.get(Admin, admin_id)
                if own_admin:
                    await db.delete(own_admin)
                    await db.commit()
                remaining = await db.scalar(select(func.count()).select_from(FaqEntry))
            result["temporary_admin_removed"] = True
            if operation == "cleanup":
                result["faq_baseline_restored"] = (
                    remaining == manifest["baseline_faq_count"]
                )
                if not result["faq_baseline_restored"]:
                    raise RuntimeError("FAQ baseline count differs")
    result["budget_after"] = BudgetStore(
        Path("/reports/release470-budget.sqlite3")
    ).snapshot()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation", choices=("create", "update", "cleanup"), required=True
    )
    args = parser.parse_args()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    logging.disable(logging.CRITICAL)
    try:
        print(json.dumps(asyncio.run(operate(args.operation)), indent=2))
    except BaseException as error:
        print(json.dumps({"failed": type(error).__name__}))
        raise SystemExit(1) from None
