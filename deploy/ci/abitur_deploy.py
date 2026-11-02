#!/usr/bin/python3
"""Restricted NSU deployment boundary; installed root-owned, never run from a checkout."""

import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOS = {"backend": "abitur-bot", "web": "abitur-web"}
SERVICES = {"backend": ("api", "bot", "migrate", "permissions"), "web": ("web",)}
PINS = Path("/opt/abitur/images.yaml")
RECEIPT = Path("/opt/abitur/deploy-receipts.json")
FAILURE = Path("/opt/abitur/deploy-failure.json")
LOCK = Path("/run/lock/abitur-deploy.lock")
COMPOSE = "/usr/local/bin/abitur-nsu-compose"
TARGETS_FILE = Path("/etc/abitur-instances/deploy-targets.json")
HEALTH_TIMEOUT = 90
HEALTH_INTERVAL = 3


@dataclass(frozen=True)
class Target:
    name: str
    api_port: int
    web_port: int
    bot: bool

    @property
    def compose(self) -> str:
        return f"/usr/local/bin/abitur-{self.name}-compose"


DEFAULT_TARGET = Target("nsu", 18001, 13001, True)


def parse_targets(data: Any) -> tuple[Target, ...]:
    """Only root configuration selects instances; the SSH request cannot select paths."""
    if not isinstance(data, list) or not 1 <= len(data) <= 3:
        raise ValueError("Invalid deployment targets")
    targets: list[Target] = []
    ports: set[int] = set()
    names: set[str] = set()
    for item in data:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "api_port",
            "web_port",
            "bot",
        }:
            raise ValueError("Invalid deployment target fields")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{1,24}", name):
            raise ValueError("Invalid deployment target name")
        if name in names or type(item["bot"]) is not bool:
            raise ValueError("Duplicate target or invalid bot selection")
        for key in ("api_port", "web_port"):
            port = item[key]
            if type(port) is not int or not 1024 <= port <= 65535 or port in ports:
                raise ValueError("Invalid or duplicate deployment port")
            ports.add(port)
        names.add(name)
        targets.append(Target(name, item["api_port"], item["web_port"], item["bot"]))
    return tuple(targets)


def load_targets(path: Path) -> tuple[Target, ...]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return (DEFAULT_TARGET,)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError("Deployment targets must be a root-owned protected regular file")
    if info.st_size > 4096:
        raise ValueError("Deployment targets file too large")
    return parse_targets(json.loads(path.read_text()))


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
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]{0,99}(?:\[bot\])?", data["registry_username"]
    ):
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


class ProcessFailure(RuntimeError):
    def __init__(self, returncode: int) -> None:
        super().__init__("Deployment subprocess failed")
        self.returncode = returncode


class Host:
    targets: tuple[Target, ...] = (DEFAULT_TARGET,)

    def __init__(self, config: str) -> None:
        self.env = {**os.environ, "DOCKER_CONFIG": config}
        self.stage = "request"
        self.targets = load_targets(TARGETS_FILE)

    def run(self, *args: str, stdin: str | None = None) -> str:
        if args[0] == "docker":
            self.stage = {
                "login": "registry-login",
                "pull": "image-pull",
                "image": "revision-check",
                "inspect": "container-state",
            }.get(args[1], "docker")
        elif args[0] in {target.compose for target in self.targets}:
            self.stage = (
                ("migration" if args[-1] == "migrate" else "permissions")
                if args[1] == "run"
                else "services"
            )
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
            raise ProcessFailure(result.returncode)
        return result.stdout.strip()

    def current_sha(self, component: str, token: str) -> str:
        self.stage = "main-freshness"
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
        for target in self.targets:
            self.healthy_target(component, target)

    def healthy_target(self, component: str, target: Target) -> None:
        self.stage = "health"
        url = (
            f"http://127.0.0.1:{target.api_port}/api/v1/university/profile"
            if component == "backend"
            else f"http://127.0.0.1:{target.web_port}/login"
        )
        deadline = time.monotonic() + HEALTH_TIMEOUT
        stable = None
        consecutive = 0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status != 200:
                        raise RuntimeError("Unhealthy HTTP status")
                if component == "backend" and target.bot:
                    container = self.run(target.compose, "ps", "-q", "bot")
                    if not container:
                        raise RuntimeError("Bot absent")
                    state = json.loads(self.run("docker", "inspect", container))[0]
                    if not state["State"]["Running"] or state["State"].get("Restarting"):
                        raise RuntimeError("Bot not running")
                    observation = (state["Id"], state["RestartCount"])
                    consecutive = consecutive + 1 if observation == stable else 1
                    stable = observation
                    if consecutive < 4:
                        time.sleep(HEALTH_INTERVAL)
                        continue
                return
            except (OSError, ValueError, RuntimeError):
                consecutive = 0
                time.sleep(HEALTH_INTERVAL)
        raise RuntimeError("Health deadline exceeded")


def start(host: Host, component: str, *, migrate: bool) -> None:
    if component == "backend" and migrate:
        for target in host.targets:
            for service in ("permissions", "migrate"):
                host.run(
                    target.compose, "run", "--rm", "--no-deps", "--pull", "never", service
                )
    for target in host.targets:
        services = ("api", "bot") if target.bot else ("api",)
        if component == "web":
            services = ("web",)
        host.run(
            target.compose,
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--no-deps",
            *services,
        )
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
    receipts = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    prior_receipt = receipts.get(component, {})
    targets = [asdict(target) for target in host.targets]
    same_pin = all(
        pins["services"][s]["image"] == data["image"] for s in SERVICES[component]
    )
    if (
        same_pin
        and prior_receipt.get("sha") == data["sha"]
        and prior_receipt.get("image") == data["image"]
        and prior_receipt.get("targets") == targets
    ):
        try:
            host.healthy(component)
        except Exception:
            pass
        else:
            return "Already deployed and healthy"
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
        failed_stage = getattr(host, "stage", "deploy")
        atomic_json(PINS, previous)
        start(host, component, migrate=False)
        host.stage = failed_stage
        raise
    receipts[component] = {
        "sha": data["sha"],
        "image": data["image"],
        "previous_image": (
            prior_receipt["previous_image"]
            if same_pin and prior_receipt.get("image") == data["image"]
            else previous["services"][SERVICES[component][0]]["image"]
        ),
        "completed_at": int(time.time()),
        "instances": [target.name for target in host.targets],
        "targets": targets,
    }
    atomic_json(RECEIPT, receipts)
    return "Deployment healthy"


def main() -> int:
    host = None
    data = None
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
            host = Host(config)
            result = deploy(sys.argv[1], data, host)
        try:
            print(result, flush=True)
        except BrokenPipeError:
            sys.stdout = open(os.devnull, "w")
        return 0
    except Exception as error:
        # Only fixed stages, exception class names and numeric codes leave root.
        detail = {
            "stage": getattr(host, "stage", "request"),
            "error": type(error).__name__,
            "returncode": error.returncode if isinstance(error, ProcessFailure) else None,
            "failed_at": int(time.time()),
        }
        if data is not None:
            detail.update({"sha": data["sha"], "image": data["image"]})
            try:
                atomic_json(FAILURE, detail)
            except OSError:
                pass
        print("Deployment failed: " + json.dumps(detail), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
