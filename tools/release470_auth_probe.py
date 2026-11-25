"""Bounded HTTP/DB auth checks for the isolated #470 primary, without providers.

Run only through the reviewed eval470 runner. No lifespan, bot, models or indexing.
Temporary accounts are removed in finally; credentials never leave this process.
"""

import asyncio
import json
import logging
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


async def probe() -> dict[str, Any]:
    """Exercise real HTTP dependencies and commits against the fixed test database."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["DB_ECHO"] = "false"
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, "/app")

    import httpx
    from sqlalchemy import delete, func, select, text, update

    from api.auth.security import hash_password
    from api.main import app
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.models import Admin, AdminRole, InviteCode

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected database target")
    if os.environ.get("UNIVERSITY_PROFILE") != "nsu":
        raise RuntimeError("Unexpected university profile")
    engine.echo = False
    prefix = "release470_" + secrets.token_hex(8)
    password = secrets.token_urlsafe(24)
    checks: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    tokens: dict[str, str] = {}
    result: dict[str, Any] = {
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "real app ASGI routes + isolated PostgreSQL; lifespan not started",
        "provider_calls": 0,
        "messages_sent": 0,
        "checks": checks,
        "cleanup_verified": False,
    }

    def record(name: str, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "passed": actual == expected,
            }
        )

    try:
        async with AsyncSessionLocal() as db:
            if await db.scalar(text("select current_database()")) != "eval470":
                raise RuntimeError("Database identity check failed")
            # Refuse writes unless the operator's sanitized baseline is still intact.
            for table in (
                '"user"',
                "message",
                "message_logs",
                "answer_feedback",
                "feedback_report",
            ):
                # Table names are fixed literals, never user input.
                if await db.scalar(text(f"select count(*) from {table}")) != 0:
                    raise RuntimeError("Expected sanitized empty user data")
            result["baseline_admin_count"] = await db.scalar(
                select(func.count()).select_from(Admin)
            )
            for role in ("superadmin", "admin", "viewer"):
                account = Admin(
                    username=f"{prefix}_{role}",
                    password_hash=hash_password(password),
                    role=AdminRole(role),
                    is_active=True,
                )
                db.add(account)
                await db.flush()
                ids[role] = account.id
            await db.commit()

        # ASGITransport deliberately does not execute the production lifespan:
        # no import queue, statistics worker, matcher refresh or provider setup.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://isolated.invalid",
            timeout=10,
        ) as client:
            for role in ids:
                response = await client.post(
                    "/api/v1/auth/login",
                    json={"username": f"{prefix}_{role}", "password": password},
                )
                record(f"login_{role}", response.status_code, 200)
                if response.status_code != 200:
                    raise RuntimeError("Test login failed")
                tokens[role] = response.json()["access_token"]

            async def request(
                name: str,
                method: str,
                path: str,
                expected: int,
                role: str | None = None,
                **kwargs: Any,
            ) -> httpx.Response:
                headers = {"Authorization": f"Bearer {tokens[role]}"} if role else {}
                response = await client.request(method, path, headers=headers, **kwargs)
                record(name, response.status_code, expected)
                return response

            await request("anonymous_me", "GET", "/api/v1/auth/me", 401)
            await request("anonymous_faq", "GET", "/api/v1/faq", 401)
            await request("viewer_faq_denied", "GET", "/api/v1/faq", 403, "viewer")
            await request("admin_faq_read", "GET", "/api/v1/faq", 200, "admin")
            await request(
                "viewer_admin_list_denied", "GET", "/api/v1/auth/admins", 403, "viewer"
            )
            await request(
                "admin_admin_list_denied", "GET", "/api/v1/auth/admins", 403, "admin"
            )
            await request(
                "superadmin_admin_list", "GET", "/api/v1/auth/admins", 200, "superadmin"
            )
            await request("viewer_identity", "GET", "/api/v1/auth/me", 200, "viewer")
            await request("viewer_topics_read", "GET", "/api/v1/topics/", 200, "viewer")
            wrong = await client.get(
                "/api/v1/auth/me", headers={"Authorization": "Bearer invalid.jwt.value"}
            )
            record("invalid_jwt", wrong.status_code, 401)
            await request(
                "wrong_password",
                "POST",
                "/api/v1/auth/login",
                401,
                json={"username": f"{prefix}_viewer", "password": "wrong-password"},
            )
            await request(
                "register_without_invite",
                "POST",
                "/api/v1/auth/register",
                400,
                json={"username": f"{prefix}_noinvite", "password": password},
            )
            await request(
                "admin_cannot_invite",
                "POST",
                "/api/v1/auth/invite",
                403,
                "admin",
                json={"role": "viewer"},
            )
            invite = await request(
                "superadmin_invite",
                "POST",
                "/api/v1/auth/invite",
                201,
                "superadmin",
                json={"role": "viewer"},
            )
            if invite.status_code != 201:
                raise RuntimeError("Test invitation failed")
            registrations = await asyncio.gather(
                *[
                    client.post(
                        "/api/v1/auth/register",
                        json={
                            "username": f"{prefix}_race_{i}",
                            "password": password,
                            "invite_code": invite.json()["code"],
                        },
                    )
                    for i in range(2)
                ]
            )
            record(
                "single_use_invite_race",
                sorted(r.status_code for r in registrations),
                [201, 400],
            )
            for response in registrations:
                if response.status_code == 201:
                    record("invite_role_preserved", response.json()["role"], "viewer")
            await request(
                "demote_admin",
                "PATCH",
                f"/api/v1/auth/admins/{ids['admin']}/role",
                200,
                "superadmin",
                json={"role": "viewer"},
            )
            await request("old_jwt_obeys_new_role", "GET", "/api/v1/faq", 403, "admin")
            await request(
                "deactivate_viewer",
                "PATCH",
                f"/api/v1/auth/admins/{ids['viewer']}/deactivate",
                200,
                "superadmin",
            )
            await request(
                "old_jwt_inactive_rejected", "GET", "/api/v1/auth/me", 401, "viewer"
            )
            await request(
                "inactive_login_rejected",
                "POST",
                "/api/v1/auth/login",
                401,
                json={"username": f"{prefix}_viewer", "password": password},
            )
    finally:
        async with AsyncSessionLocal() as db:
            # Exact random prefix belongs to this execution only. Delete invitations
            # first, then self-references, then accounts; leave seeded admin intact.
            own = select(Admin.id).where(Admin.username.startswith(prefix + "_"))
            await db.execute(delete(InviteCode).where(InviteCode.created_by_id.in_(own)))
            await db.execute(
                update(Admin).where(Admin.id.in_(own)).values(created_by_id=None)
            )
            await db.execute(delete(Admin).where(Admin.username.startswith(prefix + "_")))
            await db.commit()
            remaining = await db.scalar(
                select(func.count())
                .select_from(Admin)
                .where(Admin.username.startswith(prefix + "_"))
            )
            result["cleanup_verified"] = remaining == 0
            result["final_admin_count"] = await db.scalar(
                select(func.count()).select_from(Admin)
            )
        await engine.dispose()
    result["passed"] = (
        bool(checks) and all(c["passed"] for c in checks) and result["cleanup_verified"]
    )
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--isolated-primary"] or not Path("/evaluation").is_dir():
        raise SystemExit("Run only in the approved isolated primary worker")
    try:
        report = asyncio.run(asyncio.wait_for(probe(), timeout=150))
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)
    except Exception as exc:
        # Never print DB errors, request bodies, generated tokens or credentials.
        print(json.dumps({"probe_error": type(exc).__name__}))
        raise SystemExit(1) from None
