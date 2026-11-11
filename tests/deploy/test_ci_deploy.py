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


@pytest.mark.parametrize(
    "username", ["github-actions[bot]", "dependabot[bot]", "Marat-nsu"]
)
def test_valid_registry_actor(username: str) -> None:
    data = payload()
    data["registry_username"] = username
    assert d.validate("backend", data) == data


def test_healthy_exact_retry_preserves_rollback_receipt(state: dict) -> None:
    d.deploy("backend", payload(), FakeHost())
    original = d.RECEIPT.read_text()
    retry = FakeHost()
    assert d.deploy("backend", payload(), retry) == "Already deployed and healthy"
    assert d.RECEIPT.read_text() == original
    assert retry.calls == [("freshness",), ("health", "backend")]


def test_unhealthy_retry_repairs_without_losing_previous_image(state: dict) -> None:
    d.deploy("backend", payload(), FakeHost())
    host = FakeHost()
    host.health_failures = 1
    # This retry legitimately pulls with the new pin already present.
    original_run = host.run

    def retry_run(*args: str, stdin: str | None = None) -> str:
        if args[:2] == ("docker", "pull"):
            host.calls.append(args)
            return ""
        return original_run(*args, stdin=stdin)

    host.run = retry_run
    assert d.deploy("backend", payload(), host) == "Deployment healthy"
    assert json.loads(d.RECEIPT.read_text())["backend"]["previous_image"] == "old-backend"


def test_failure_receipt_and_diagnostics_never_include_secret(
    state: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    import io
    import signal
    import sys

    class FailedHost(FakeHost):
        def __init__(self, config: str) -> None:
            super().__init__()
            self.stage = "image-pull"

        def run(self, *args: str, stdin: str | None = None) -> str:
            raise d.ProcessFailure(17)

    monkeypatch.setattr(d, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(d, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(d, "Host", FailedHost)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["deploy", "backend"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload())))
    assert d.main() == 1
    output = capsys.readouterr().err
    assert '"stage": "image-pull"' in output
    assert '"returncode": 17' in output
    assert "secret" not in output + d.FAILURE.read_text()
    assert json.loads(d.PINS.read_text()) == state


PREV = "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "c" * 64
CUR = "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "d" * 64


@pytest.fixture
def rolled_state(state: dict) -> dict:
    """Успешный деплой уже прошёл: pins на текущем образе, receipt помнит предыдущий."""
    pins = json.loads(d.PINS.read_text())
    for service in d.SERVICES["backend"]:
        pins["services"][service] = {"image": CUR}
    d.atomic_json(d.PINS, pins)
    d.atomic_json(
        d.RECEIPT,
        {
            "backend": {
                "sha": SHA,
                "image": CUR,
                "previous_image": PREV,
                "completed_at": 1,
                "instances": ["nsu"],
                "targets": [
                    {"name": "nsu", "api_port": 18001, "web_port": 13001, "bot": True}
                ],
            }
        },
    )
    return pins


def test_rollback_pins_previous_image_without_migrations(rolled_state: dict) -> None:
    host = FakeHost()

    assert d.rollback("backend", host) == "Rollback healthy"

    pins = json.loads(d.PINS.read_text())
    for service in d.SERVICES["backend"]:
        assert pins["services"][service] == {"image": PREV}
    # Чужой компонент не тронут.
    assert pins["services"]["web"] == rolled_state["services"]["web"]
    # Миграции не запускаются и не откатываются.
    assert not any("migrate" in call or "permissions" in call for call in host.calls)
    assert ("docker", "login") not in host.calls
    assert ("docker", "pull", PREV) not in host.calls
    receipt = json.loads(d.RECEIPT.read_text())["backend"]
    assert receipt["image"] == PREV
    assert receipt["previous_image"] == CUR
    assert receipt["manual_rollback"] is True


def test_rollback_without_receipt_fails_closed(state: dict) -> None:
    host = FakeHost()

    with pytest.raises(ValueError, match="previous image"):
        d.rollback("backend", host)

    assert json.loads(d.PINS.read_text()) == state
    assert not any("up" in call for call in host.calls)


def test_rollback_health_failure_restores_pin(rolled_state: dict) -> None:
    host = FakeHost()
    host.health_failures = 1

    with pytest.raises(RuntimeError):
        d.rollback("backend", host)

    pins = json.loads(d.PINS.read_text())
    for service in d.SERVICES["backend"]:
        assert pins["services"][service] == {"image": CUR}
    # Откат отката: сервисы подняты повторно на прежнем образе.
    assert sum("up" in call for call in host.calls) == 2
    receipt = json.loads(d.RECEIPT.read_text())["backend"]
    assert receipt["image"] == CUR


def test_rollback_same_image_is_noop(rolled_state: dict) -> None:
    pins = json.loads(d.PINS.read_text())
    for service in d.SERVICES["backend"]:
        pins["services"][service] = {"image": PREV}
    d.atomic_json(d.PINS, pins)

    host = FakeHost()
    assert d.rollback("backend", host) == "Already rolled back"
    assert not any("up" in call for call in host.calls)
