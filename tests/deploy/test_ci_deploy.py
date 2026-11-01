import json
from pathlib import Path

import pytest

from deploy.ci import abitur_deploy as d

SHA = "a" * 40
IMAGE = "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "b" * 64


def payload() -> dict[str, str]:
    return dict(image=IMAGE, sha=SHA, registry_username="actor", registry_token="secret")


class FakeHost(d.Host):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fresh = [SHA, SHA]
        self.revision = SHA
        self.fail = ""
        self.health_failures = 0

    def current_sha(self, component: str, token: str) -> str:
        self.calls.append(("freshness",))
        return self.fresh.pop(0)

    def run(self, *args: str, stdin: str | None = None) -> str:
        self.calls.append(args)
        assert "secret" not in args
        if self.fail and self.fail in args:
            raise RuntimeError("fake failure")
        if "pull" in args:
            assert "old-backend" in d.PINS.read_text()
        if "login" in args:
            assert stdin == "secret"
        if "inspect" in args:
            return self.revision
        return ""

    def healthy(self, component: str) -> None:
        self.calls.append(("health", component))
        if self.health_failures:
            self.health_failures -= 1
            raise RuntimeError("unhealthy")


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(d, "PINS", tmp_path / "images.yaml")
    monkeypatch.setattr(d, "RECEIPT", tmp_path / "receipts.json")
    pins = {
        "services": {
            s: {"image": "old-" + c}
            for c, services in d.SERVICES.items()
            for s in services
        }
    }
    d.atomic_json(d.PINS, pins)
    return pins


@pytest.mark.parametrize(
    "key,value",
    [
        ("sha", "main"),
        ("image", "ghcr.io/evil/image@sha256:" + "b" * 64),
        ("image", IMAGE + ";sh"),
        ("registry_token", "bad\ntoken"),
        ("registry_username", "--password=oops"),
    ],
)
def test_rejects_invalid_request(key: str, value: str) -> None:
    data = payload()
    data[key] = value
    with pytest.raises(ValueError):
        d.validate("backend", data)


def test_success_order_and_other_component_preserved(state: dict) -> None:
    host = FakeHost()
    assert d.deploy("backend", payload(), host) == "Deployment healthy"
    pins = json.loads(d.PINS.read_text())
    assert pins["services"]["web"] == state["services"]["web"]
    assert pins["services"]["api"]["image"] == IMAGE
    commands = [" ".join(call) for call in host.calls]
    assert commands[0] == "freshness"
    assert next(i for i, c in enumerate(commands) if "docker pull" in c) < next(
        i for i, c in enumerate(commands) if "run --rm" in c
    )
    assert commands[-1] == "health backend"
    assert "previous_image" in d.RECEIPT.read_text()
    assert "secret" not in d.RECEIPT.read_text()


@pytest.mark.parametrize("after_pull", [False, True])
def test_stale_revision_never_changes_pins(state: dict, after_pull: bool) -> None:
    host = FakeHost()
    host.fresh = [SHA, "c" * 40] if after_pull else ["c" * 40]
    assert "Skipped" in d.deploy("backend", payload(), host)
    assert json.loads(d.PINS.read_text()) == state
    assert not any("up" in call for call in host.calls)


@pytest.mark.parametrize("failure", ["pull", "revision", "health", "migrate"])
def test_failures_preserve_or_restore_affected_pin(state: dict, failure: str) -> None:
    host = FakeHost()
    if failure == "revision":
        host.revision = "wrong"
    elif failure == "health":
        host.health_failures = 1
    else:
        host.fail = failure
    with pytest.raises(Exception):
        d.deploy("backend", payload(), host)
    assert json.loads(d.PINS.read_text()) == state
    assert not d.RECEIPT.exists()
    if failure in ("health", "migrate"):
        assert sum("migrate" in call for call in host.calls) == 1


def test_wrapper_rejects_extra_command(tmp_path: Path) -> None:
    import os
    import subprocess

    wrapper = Path("deploy/ci/abitur-ci-ssh")
    for command in ("", "deploy; id", "deploy ", "sh"):
        result = subprocess.run(
            ["sh", str(wrapper), "backend"],
            env={**os.environ, "SSH_ORIGINAL_COMMAND": command},
        )
        assert result.returncode == 1


def test_web_only_starts_web_and_preserves_backend(state: dict) -> None:
    host = FakeHost()
    data = payload()
    data["image"] = IMAGE.replace("abitur-bot@", "abitur-web@")
    d.deploy("web", d.validate("web", data), host)
    pins = json.loads(d.PINS.read_text())
    for service in d.SERVICES["backend"]:
        assert pins["services"][service] == state["services"][service]
    assert not any("migrate" in call or "permissions" in call for call in host.calls)
    assert (
        d.COMPOSE,
        "up",
        "-d",
        "--no-build",
        "--pull",
        "never",
        "--no-deps",
        "web",
    ) in host.calls


def test_process_boundary_captures_output_and_passes_token_only_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import subprocess

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 1, "secret", "secret")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="^Deployment subprocess failed$"):
        d.Host(str(tmp_path)).run("docker", "login", "--password-stdin", stdin="secret")
    args, kwargs = calls[0]
    assert "secret" not in args
    assert kwargs["input"] == "secret"
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 240
    assert kwargs["env"]["DOCKER_CONFIG"] == str(tmp_path)


def test_main_locks_before_freshness_and_cleans_private_registry_config(
    state: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fcntl
    import io
    import signal
    import sys

    lock_path = tmp_path / "shared.lock"
    monkeypatch.setattr(d, "LOCK", lock_path)
    monkeypatch.setattr(sys, "argv", ["abitur-deploy", "backend"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload())))
    signals = []
    configs = []
    monkeypatch.setattr(
        signal, "signal", lambda sig, action: signals.append((sig, action))
    )

    class LockedHost(FakeHost):
        def __init__(self, config: str) -> None:
            super().__init__()
            configs.append(Path(config))
            assert Path(config).stat().st_mode & 0o777 == 0o700

        def current_sha(self, component: str, token: str) -> str:
            assert (signal.SIGHUP, signal.SIG_IGN) in signals
            assert (signal.SIGTERM, signal.SIG_IGN) in signals
            with lock_path.open("w") as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return super().current_sha(component, token)

    monkeypatch.setattr(d, "Host", LockedHost)
    assert d.main() == 0
    assert configs and all(not path.exists() for path in configs)
    assert "secret" not in d.RECEIPT.read_text()
