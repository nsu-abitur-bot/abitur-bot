"""Destructive lifecycle checks for two disposable demo projects (not real pilots)."""

import argparse
import json
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

COMPOSE = Path(__file__).with_name("compose.yaml").resolve()
FIXTURE = """
import asyncio
import json
from pathlib import Path
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document, Message, User
async def main():
    async with AsyncSessionLocal() as s:
        u = User()
        s.add(u)
        await s.flush()
        s.add(Message(user_id=u.user_id, session_id="demo-probe",
                      user_text="MARKER", bot_response="MARKER"))
        s.add(Document(id="MARKER", title="MARKER",
                       graph_id="abitur_kb", rag_doc_id="MARKER"))
        await s.commit()
    root = Path("/app/data/lightrag/demo-probe")
    root.mkdir(parents=True, exist_ok=True)
    (root / "MARKER").write_text("MARKER")
    kb = Path("/app/data/lightrag/abitur_kb")
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "kv_store_full_docs.json").write_text(
        json.dumps({"MARKER": {"content": "MARKER"}}))
    Path("/app/logs/MARKER").write_text("MARKER")
asyncio.run(main())
"""


class Stand:
    def __init__(self, env: Path) -> None:
        self.env = str(env.resolve())
        self.values = dict(
            line.split("=", 1)
            for line in env.read_text().splitlines()
            if line and not line.startswith("#")
        )
        self.token = ""

    def compose(self, *args: str) -> str:
        result = subprocess.run(
            ["docker", "compose", "--env-file", self.env, "-f", str(COMPOSE), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def request(self, path: str, body: dict | None = None) -> tuple[int, object]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.values["API_ORIGIN"] + "/api/v1" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, {}

    def wait(self) -> None:
        for _ in range(90):
            try:
                if self.request("/auth/me")[0] in (200, 401):
                    return
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError("API did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    a, b = Stand(args.first), Stand(args.second)
    assert a.values["INSTANCE_NAME"] != b.values["INSTANCE_NAME"]
    for key in ("SECRET_KEY", "SETTINGS_ENCRYPTION_KEY", "DB_PASSWORD", "REDIS_PASSWORD"):
        assert a.values[key] != b.values[key], key
    marker = "probe-" + secrets.token_hex(8)
    password = secrets.token_hex(16)
    for stand in (a, b):
        stand.wait()
        with urllib.request.urlopen(stand.values["WEB_ORIGIN"] + "/login") as response:
            assert response.status == 200
    assert (
        a.request("/auth/register", {"username": marker, "password": password})[0] == 201
    )
    status, login = a.request("/auth/login", {"username": marker, "password": password})
    assert status == 200 and isinstance(login, dict)
    a.token = login["access_token"]
    assert b.request("/auth/login", {"username": marker, "password": password})[0] == 401
    b.token = a.token
    assert b.request("/auth/me")[0] == 401
    b.token = ""
    assert (
        b.request("/auth/register", {"username": marker + "b", "password": password})[0]
        == 201
    )
    _, login_b = b.request(
        "/auth/login", {"username": marker + "b", "password": password}
    )
    assert isinstance(login_b, dict)
    b.token = login_b["access_token"]
    a.compose("exec", "-T", "api", "python", "-c", FIXTURE.replace("MARKER", marker))
    a.compose("exec", "-T", "redis", "sh", "-c", f"redis-cli SET demo-probe {marker}")

    def check() -> None:
        a.wait()
        b.wait()
        assert marker in json.dumps(a.request("/messages")[1])
        assert marker not in json.dumps(b.request("/messages")[1])
        assert a.request(f"/rag/docs/{marker}/content") == (
            200,
            {"id": marker, "content": marker},
        )
        assert b.request(f"/rag/docs/{marker}/content")[0] == 404
        for stand, expected in ((a, "1"), (b, "0")):
            assert (
                stand.compose(
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-U",
                    "abitur",
                    "-d",
                    "abitur",
                    "-Atc",
                    f"SELECT count(*) FROM document WHERE id='{marker}'",
                )
                == expected
            )
            files = stand.compose(
                "exec",
                "-T",
                "api",
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    f'print(Path("/app/data/lightrag/demo-probe/{marker}").exists(), '
                    f'Path("/app/logs/{marker}").exists())'
                ),
            )
            assert files == ("True True" if stand is a else "False False")
        assert (
            a.compose("exec", "-T", "redis", "sh", "-c", "redis-cli GET demo-probe")
            == marker
        )
        assert (
            b.compose("exec", "-T", "redis", "sh", "-c", "redis-cli GET demo-probe") == ""
        )

    check()
    second_id = b.compose("ps", "-q", "api")
    a.compose("restart", "api", "redis", "postgres", "web")
    check()
    a.compose("up", "-d", "--force-recreate")
    check()
    a.compose("down")
    assert b.compose("ps", "-q", "api") == second_id
    assert b.request("/auth/me")[0] == 200
    a.compose("up", "-d")
    check()
    assert b.compose("ps", "-q", "api") == second_id
    print("PASS: auth, documents, history, Redis, files, logs and lifecycle isolation")


if __name__ == "__main__":
    main()
