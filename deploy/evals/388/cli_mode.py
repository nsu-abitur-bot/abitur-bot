"""Root-owned extension for isolated #388 CLI jobs, without an API server."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

BASE = Path("/etc/abitur-evals/388")
DOCKER = "/usr/bin/docker"
LABEL = "org.abitur.eval388.cli-run"
MIN_MEMORY_KIB = 1100 * 1024
TIMEOUT_SECONDS = 300
OVERLAY = {
    "services": {
        "api": {
            "read_only": True,
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=32m"],
            "security_opt": ["no-new-privileges:true"],
            "healthcheck": {"disable": True},
            "restart": "no",
        }
    }
}


class Refused(RuntimeError):
    pass


def require(value: object) -> None:
    if not value:
        raise Refused()


def validate_script(script: str) -> None:
    require(re.fullmatch(r"/evaluation/[a-z][a-z0-9_]{0,80}\.py", script))


def compose(variant: str) -> list[str]:
    require(variant in {"with-kb", "empty-kb"})
    return [
        DOCKER,
        "compose",
        "--project-name",
        f"abitur-eval388-{variant}",
        "--project-directory",
        str(BASE / variant),
        "-f",
        str(BASE / variant / "compose.json"),
        "-f",
        str(BASE / "cli-overlay.json"),
    ]


def worker_command(variant: str, script: str, args: list[str], run_id: str) -> list[str]:
    validate_script(script)
    require(re.fullmatch(r"[0-9a-f]{32}", run_id))
    require(len(args) <= 32 and all(len(arg) <= 1024 for arg in args))
    return compose(variant) + [
        "run",
        "--rm",
        "--no-deps",
        "--pull",
        "never",
        "-T",
        "--name",
        f"abitur-eval388-{variant}-cli",
        "--label",
        f"{LABEL}={run_id}",
        "--user",
        "1000:1000",
        "--cap-drop",
        "ALL",
        "--entrypoint",
        "/app/.venv/bin/python",
        "--env",
        "PYTHON_DOTENV_DISABLED=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/evaluation:/app",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MALLOC_ARENA_MAX=2",
        "--env",
        f"EVAL388_RUN_ID={run_id}",
        "api",
        script,
        *args,
    ]


def execute(command: list[str], *, timeout: int = 40) -> str:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )
    require(result.returncode == 0)
    return result.stdout.strip()


def project_containers(variant: str) -> str:
    return execute(
        [
            DOCKER,
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project=abitur-eval388-{variant}",
        ]
    )


def available_memory() -> int:
    return next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def memory_bytes(value: str | int) -> int:
    if type(value) is int:
        return value
    match = re.fullmatch(r"([0-9]+)([kKmMgG])(?:[bB])?", str(value))
    if match is None:
        raise Refused()
    return int(match[1]) * 1024 ** ({"k": 1, "m": 2, "g": 3}[match[2].lower()])


def validate_config(variant: str) -> None:
    # Read only fixed root-owned deployment inputs. Never print their contents.
    path = BASE / variant / "compose.json"
    config = json.loads(path.read_text())
    require(json.loads((BASE / "cli-overlay.json").read_text()) == OVERLAY)
    version = json.loads((BASE / "version.json").read_text())
    api = config["services"]["api"]
    require(config.get("name") == f"abitur-eval388-{variant}")
    require(api.get("image") == version["image"])
    require(
        re.fullmatch(
            r"ghcr\.io/nsu-abitur-bot/abitur-bot@sha256:[0-9a-f]{64}", version["image"]
        )
    )
    require(0 < memory_bytes(api["mem_limit"]) <= 640 * 1024**2)
    require(
        memory_bytes(api["mem_limit"])
        <= memory_bytes(api["memswap_limit"])
        <= 896 * 1024**2
    )
    require(0 < float(api["cpus"]) <= 0.75)
    require(not api.get("pre_start") and not api.get("post_start"))
    require(
        not api.get("privileged") and not api.get("cap_add") and not api.get("devices")
    )
    require(api.get("network_mode") != "host" and api.get("pid") != "host")
    for service, maximum in (("postgres", 128), ("redis", 32)):
        spec = config["services"][service]
        require(0 < memory_bytes(spec["mem_limit"]) <= maximum * 1024**2)
        require(not spec.get("pre_start") and not spec.get("post_start"))


def stop_owned_worker(name: str, run_id: str) -> None:
    ids = execute([DOCKER, "ps", "-aq", "--filter", f"name=^/{name}$"])
    if not ids:
        return
    require(re.fullmatch(r"[0-9a-f]{12,64}", ids))
    marker = execute(
        [
            DOCKER,
            "inspect",
            "--format",
            '{{ index .Config.Labels "' + LABEL + '" }}',
            ids,
        ]
    )
    require(marker == run_id)
    execute([DOCKER, "rm", "-f", ids])


def run_job(
    variant: str,
    script: str,
    args: list[str],
    *,
    before_cleanup: Callable[[], None] | None = None,
) -> dict:
    validate_script(script)
    validate_config(variant)
    command = worker_command(variant, script, args, uuid.uuid4().hex)
    run_id = command[command.index("--label") + 1].split("=", 1)[1]
    require(not project_containers("with-kb") and not project_containers("empty-kb"))
    require(
        not execute(
            [
                DOCKER,
                "ps",
                "-aq",
                "--filter",
                f"name=^/abitur-eval388-{variant}-cli$",
            ]
        )
    )
    before = available_memory()
    require(before >= MIN_MEMORY_KIB)
    disk = os.statvfs("/")
    require(disk.f_bavail * disk.f_frsize >= 5 * 1024**3)
    execute(["/usr/local/sbin/abitur-eval388-firewall"])
    started = False
    cleanup_ok = True
    code = None
    timed_out = False
    error_type = None
    began = time.monotonic()
    try:
        started = True
        # --no-deps prevents migrations, permissions jobs, API and bots starting.
        execute(
            compose(variant)
            + [
                "up",
                "-d",
                "--no-deps",
                "--no-build",
                "--pull",
                "never",
                "--wait",
                "--wait-timeout",
                "40",
                "postgres",
                "redis",
            ],
            timeout=60,
        )
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TIMEOUT_SECONDS,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    except BaseException as exc:
        error_type = type(exc).__name__
    finally:
        # A failed/partial start also belongs to this invocation. Never delete volumes.
        if started:
            if before_cleanup is not None:
                before_cleanup()
            try:
                stop_owned_worker(f"abitur-eval388-{variant}-cli", run_id)
                execute(compose(variant) + ["down", "--timeout", "20"], timeout=45)
                require(not project_containers(variant))
            except (Refused, subprocess.SubprocessError, OSError):
                cleanup_ok = False
    return {
        "run_id": run_id,
        "variant": variant,
        "worker_exit_code": code,
        "timed_out": timed_out,
        "error_type": error_type,
        "cleanup_ok": cleanup_ok,
        "elapsed_seconds": round(time.monotonic() - began, 2),
        "host_available_before_kib": before,
        "host_available_after_kib": available_memory(),
        "api_server_started": False,
        "status": "completed"
        if code == 0 and cleanup_ok and not timed_out and not error_type
        else "failed",
    }


def main() -> int:
    # Called only by the existing root wrapper. It already clears the environment.
    import fcntl

    require(os.geteuid() == 0)
    require(len(sys.argv) >= 3)
    variant, script, *args = sys.argv[1:]
    validate_script(script)
    compose(variant)
    for parent in (Path(__file__), Path(__file__).parent):
        require(not parent.is_symlink())
        stat = parent.stat()
        require(stat.st_uid == 0 and not stat.st_mode & 0o022)

    def ignore_termination() -> None:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)

    def interrupt(signum: int, _frame: object) -> None:
        ignore_termination()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with Path("/run/lock/abitur-eval388.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_job(variant, script, args, before_cleanup=ignore_termination)
    print(json.dumps(result))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Refused, subprocess.SubprocessError, OSError, ValueError, KeyError):
        # Docker output and exception messages may contain environment values.
        print(json.dumps({"status": "refused_or_failed", "details_suppressed": True}))
        raise SystemExit(1)
