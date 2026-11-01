#!/usr/bin/python3
"""Restricted NSU deployment boundary; installed root-owned, never run from a checkout."""

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

REPOS = {"backend": "abitur-bot", "web": "abitur-web"}
SERVICES = {"backend": ("api", "bot", "migrate", "permissions"), "web": ("web",)}
PINS = Path("/opt/abitur/images.yaml")
RECEIPT = Path("/opt/abitur/deploy-receipts.json")
LOCK = Path("/run/lock/abitur-deploy.lock")
COMPOSE = "/usr/local/bin/abitur-nsu-compose"


def validate(component: str, data: Any) -> dict[str, str]:
    if component not in REPOS or not isinstance(data, dict):
        raise ValueError("Invalid deployment request")
    if set(data) != {"image", "sha", "registry_username", "registry_token"}:
        raise ValueError("Invalid request keys")
    if not all(isinstance(v, str) and v and len(v) <= 4096 for v in data.values()):
        raise ValueError("Invalid request values")
    image = "ghcr.io/nsu-abitur-bot/" + REPOS[component]
    if not re.fullmatch(re.escape(image) + r"@sha256:[0-9a-f]{64}", data["image"]):
        raise ValueError("Invalid image digest")
    if not re.fullmatch(r"[0-9a-f]{40}", data["sha"]):
        raise ValueError("Invalid revision")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,99}", data["registry_username"]):
        raise ValueError("Invalid registry username")
    if any(c.isspace() for c in data["registry_token"]):
        raise ValueError("Invalid registry token")
    return data


def atomic_json(path: Path, data: Any) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".deploy-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Host:
    def __init__(self, config: str) -> None:
        self.env = {**os.environ, "DOCKER_CONFIG": config}

    def run(self, *args: str, stdin: str | None = None) -> str:
        # Never forward command output: docker errors may contain credentials.
        result = subprocess.run(
            args,
            input=stdin,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=240,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Deployment subprocess failed")
        return result.stdout.strip()

    def current_sha(self, component: str, token: str) -> str:
        request = urllib.request.Request(
            f"https://api.github.com/repos/nsu-abitur-bot/{REPOS[component]}"
            "/git/ref/heads/main",
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)["object"]["sha"]

    def healthy(self, component: str) -> None:
        url = (
            "http://127.0.0.1:18001/api/v1/university/profile"
            if component == "backend"
            else "http://127.0.0.1:13001/login"
        )
        deadline = time.monotonic() + 90
        stable = None
        consecutive = 0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status != 200:
                        raise RuntimeError("Unhealthy HTTP status")
                if component == "backend":
                    container = self.run(COMPOSE, "ps", "-q", "bot")
                    if not container:
                        raise RuntimeError("Bot absent")
                    state = json.loads(self.run("docker", "inspect", container))[0]
                    if not state["State"]["Running"] or state["State"].get("Restarting"):
                        raise RuntimeError("Bot not running")
                    observation = (state["Id"], state["RestartCount"])
                    consecutive = consecutive + 1 if observation == stable else 1
                    stable = observation
                    if consecutive < 4:
                        time.sleep(3)
                        continue
                return
            except (OSError, ValueError, RuntimeError):
                consecutive = 0
                time.sleep(3)
        raise RuntimeError("Health deadline exceeded")


def start(host: Host, component: str, *, migrate: bool) -> None:
    if component == "backend" and migrate:
        for service in ("permissions", "migrate"):
            host.run(COMPOSE, "run", "--rm", "--no-deps", "--pull", "never", service)
    services = ("api", "bot") if component == "backend" else ("web",)
    host.run(COMPOSE, "up", "-d", "--no-build", "--pull", "never", "--no-deps", *services)
    host.healthy(component)


def deploy(component: str, data: dict[str, str], host: Host) -> str:
    """Caller holds the shared lock; freshness is checked before any pin mutation."""
    if host.current_sha(component, data["registry_token"]) != data["sha"]:
        return "Skipped outdated main revision"
    pins = json.loads(PINS.read_text())
    expected = set(SERVICES["backend"] + SERVICES["web"])
    if set(pins) != {"services"} or set(pins["services"]) != expected:
        raise ValueError("Unexpected image overlay")
    if any(set(value) != {"image"} for value in pins["services"].values()):
        raise ValueError("Overlay must contain images only")
    previous = json.loads(json.dumps(pins))
    host.run(
        "docker",
        "login",
        "ghcr.io",
        "--username",
        data["registry_username"],
        "--password-stdin",
        stdin=data["registry_token"],
    )
    host.run("docker", "pull", data["image"])
    revision = host.run(
        "docker",
        "image",
        "inspect",
        "--format",
        '{{index .Config.Labels "org.opencontainers.image.revision"}}',
        data["image"],
    )
    if revision != data["sha"]:
        raise ValueError("Image revision differs from tested commit")
    # Pull may be slow; check again immediately before changing the running pin.
    if host.current_sha(component, data["registry_token"]) != data["sha"]:
        return "Skipped outdated main revision after pull"
    for service in SERVICES[component]:
        pins["services"][service] = {"image": data["image"]}
    atomic_json(PINS, pins)
    try:
        start(host, component, migrate=True)
    except Exception:
        atomic_json(PINS, previous)
        start(host, component, migrate=False)
        raise RuntimeError(
            "Deploy failed; previous image restored (schema not downgraded)"
        ) from None
    receipts = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    receipts[component] = {
        "sha": data["sha"],
        "image": data["image"],
        "previous_image": previous["services"][SERVICES[component][0]]["image"],
        "completed_at": int(time.time()),
    }
    atomic_json(RECEIPT, receipts)
    return "Deployment healthy"


def main() -> int:
    try:
        data = validate(
            sys.argv[1] if len(sys.argv) == 2 else "", json.loads(sys.stdin.read(20000))
        )
        # SSH disconnect or Actions cancellation must not interrupt an accepted deploy.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with (
            LOCK.open("w") as lock,
            tempfile.TemporaryDirectory(prefix="abitur-registry-") as config,
        ):
            os.chmod(config, 0o700)
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = deploy(sys.argv[1], data, Host(config))
        try:
            print(result, flush=True)
        except BrokenPipeError:
            sys.stdout = open(os.devnull, "w")
        return 0
    except Exception:
        # Do not expose exception strings, payloads, registry responses or tokens.
        print("Deployment failed; inspect root-owned state", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
