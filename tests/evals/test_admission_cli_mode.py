import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import admission_memory_probe as probe
from tools import prepare_admission_cli_wrapper as prepare

spec = importlib.util.spec_from_file_location(
    "admission_cli_mode", Path(__file__).parents[2] / "deploy/evals/388/cli_mode.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("variant", ["with-kb", "empty-kb"])
def test_worker_overrides_api_command_without_publishing_ports(variant):
    cmd = cli.worker_command(
        variant, "/evaluation/admission_memory_probe.py", ["--execute"], "a" * 32
    )
    assert cmd[-3:] == ["api", "/evaluation/admission_memory_probe.py", "--execute"]
    assert "--no-deps" in cmd and "--rm" in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "/app/.venv/bin/python"
    assert not {"--service-ports", "--publish", "--volume", "--privileged"} & set(cmd)
    assert cmd[cmd.index("--user") + 1] == "1000:1000"


@pytest.mark.parametrize(
    "script", ["-c", "/app/main.py", "/evaluation/../main.py", "/evaluation/link/code.py"]
)
def test_only_direct_evaluation_scripts_are_accepted(script):
    with pytest.raises(cli.Refused):
        cli.validate_script(script)


@pytest.mark.parametrize(
    "failure", ["none", "start", "worker", "timeout", "cleanup", "foreign", "interrupt"]
)
def test_cleanup_only_owned_worker_and_project_without_volumes(monkeypatch, failure):
    commands = []
    ps_calls = 0
    cleanup_entered = []
    monkeypatch.setattr(cli, "validate_config", lambda variant: None)
    monkeypatch.setattr(cli, "project_containers", lambda variant: "")
    monkeypatch.setattr(cli, "available_memory", lambda: 2 * cli.MIN_MEMORY_KIB)
    monkeypatch.setattr(
        cli.os,
        "statvfs",
        lambda path: SimpleNamespace(f_bavail=10**9, f_frsize=4096),
        raising=False,
    )

    def execute(cmd, **kwargs):
        nonlocal ps_calls
        commands.append(cmd)
        if "up" in cmd and failure == "start":
            raise cli.Refused()
        if "down" in cmd and failure == "cleanup":
            raise cli.Refused()
        if "ps" in cmd:
            ps_calls += 1
            return "" if ps_calls == 1 else "a" * 12
        if "inspect" in cmd:
            return "wrong-owner" if failure == "foreign" else "a" * 32
        return ""

    def worker(cmd, **kwargs):
        commands.append(cmd)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 300)
        if failure == "interrupt":
            raise SystemExit(143)
        return SimpleNamespace(returncode=1 if failure == "worker" else 0)

    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(cli.subprocess, "run", worker)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    result = cli.run_job(
        "with-kb",
        "/evaluation/admission_memory_probe.py",
        ["--execute"],
        before_cleanup=lambda: cleanup_entered.append(True),
    )
    assert (result["status"] == "completed") == (failure == "none")
    assert result["cleanup_ok"] == (failure not in {"cleanup", "foreign"})
    assert cleanup_entered == [True]
    assert any("up" in cmd and cmd[-2:] == ["postgres", "redis"] for cmd in commands)
    assert not any(
        "--volumes" in cmd or "reset" in cmd or "migrate" in cmd for cmd in commands
    )
    if failure == "foreign":
        assert not any("rm" in cmd or "down" in cmd for cmd in commands)
    else:
        assert any("rm" in cmd for cmd in commands)
        assert any("down" in cmd for cmd in commands)


@pytest.mark.parametrize("reason", ["memory", "existing"])
def test_start_guard_refuses_before_mutations(monkeypatch, reason):
    commands = []
    monkeypatch.setattr(cli, "validate_config", lambda variant: None)
    monkeypatch.setattr(
        cli,
        "project_containers",
        lambda variant: "existing" if reason == "existing" else "",
    )
    monkeypatch.setattr(cli, "available_memory", lambda: 800 * 1024)
    monkeypatch.setattr(cli, "execute", lambda cmd: commands.append(cmd) or "")
    with pytest.raises(cli.Refused):
        cli.run_job("with-kb", "/evaluation/admission_memory_probe.py", [])
    assert not any("up" in cmd or "run" in cmd or "down" in cmd for cmd in commands)


def test_wrapper_extension_preserves_existing_dispatch_and_checks_version(monkeypatch):
    original = b"safe prelude\n" + prepare.MARKER + b"existing actions\n"
    monkeypatch.setattr(prepare, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    updated = prepare.render(original)
    assert updated.replace(prepare.DISPATCH, b"") == original
    assert b"'-I', '/usr/local/libexec/abitur-eval388-cli.py'" in updated
    with pytest.raises(ValueError):
        prepare.render(original + b"changed")


def test_network_guard_allows_only_reference_stores_and_blocks_provider():
    guard = probe.NetworkGuard({("172.20.0.2", 5432), ("172.20.0.3", 6379)})
    guard.audit("socket.connect", (None, ("172.20.0.2", 5432)))
    guard.audit("socket.getaddrinfo", ("redis", 6379))
    for event, args in [
        ("socket.connect", (None, ("172.20.0.2", 443))),
        ("socket.getaddrinfo", ("api.deepseek.com", 443)),
        ("socket.connect", (None, "/var/run/docker.sock")),
    ]:
        with pytest.raises(RuntimeError):
            guard.audit(event, args)
    assert guard.blocked == 3


def test_probe_records_failure_before_emergency_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe,
        "snapshot",
        lambda: {"host_available_kib": 100 * 1024, "memory.swap.current": 5},
    )

    def exit_now(code):
        raise RuntimeError(f"test exit {code}")

    monkeypatch.setattr(probe.os, "_exit", exit_now)
    report = tmp_path / "memory.json"
    recorder = probe.Recorder(report, "a" * 32)
    with pytest.raises(RuntimeError, match="test exit 75"):
        recorder.record("import")
    assert json.loads(report.read_text())["status"] == "aborted_low_host_memory"
    with pytest.raises(FileExistsError):
        probe.Recorder(report, "a" * 32)


@pytest.mark.parametrize("change", ["none", "image", "memory", "hooks", "overlay"])
def test_root_configuration_is_checked_before_start(tmp_path, monkeypatch, change):
    monkeypatch.setattr(cli, "BASE", tmp_path)
    image = "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "a" * 64
    api: dict = {
        "image": image,
        "mem_limit": "640m",
        "memswap_limit": "896m",
        "cpus": 0.75,
    }
    if change == "image":
        api["image"] = "other-image"
    if change == "memory":
        api["mem_limit"] = "800m"
    if change == "hooks":
        api["pre_start"] = [{"command": "unexpected"}]
    folder = tmp_path / "with-kb"
    folder.mkdir()
    (folder / "compose.json").write_text(
        json.dumps(
            {
                "name": "abitur-eval388-with-kb",
                "services": {
                    "api": api,
                    "postgres": {"mem_limit": "128m"},
                    "redis": {"mem_limit": "32m"},
                },
            }
        )
    )
    (tmp_path / "version.json").write_text(json.dumps({"image": image}))
    (tmp_path / "cli-overlay.json").write_text(
        json.dumps({} if change == "overlay" else cli.OVERLAY)
    )
    if change == "none":
        cli.validate_config("with-kb")
    else:
        with pytest.raises(cli.Refused):
            cli.validate_config("with-kb")


def test_probe_uses_independent_copy_and_never_reuses_one(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in probe.FILES:
        if name != "kv_store_llm_response_cache.json":
            (source / name).write_bytes(b"fixed-source")
    target = tmp_path / "copy"
    hashes = probe.prepare_copy(source, target)
    (target / "abitur_kb" / probe.FILES[0]).write_bytes(b"changed-copy")
    assert all(probe.sha256(source / name) == value for name, value in hashes.items())
    with pytest.raises(FileExistsError):
        probe.prepare_copy(source, target)


def test_missing_source_refuses_before_creating_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "copy"
    with pytest.raises(RuntimeError):
        probe.prepare_copy(source, target)
    assert not target.exists()
