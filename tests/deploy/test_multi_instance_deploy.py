import json
from pathlib import Path

import pytest

from deploy.ci import abitur_deploy as d


def configured_targets() -> list[dict]:
    return [
        {"name": "nsu", "api_port": 18001, "web_port": 13001, "bot": True},
        {"name": "nstu", "api_port": 18002, "web_port": 13002, "bot": False},
    ]


def test_multi_instance_backend_failure_rolls_back_both_without_new_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = d.parse_targets(configured_targets())
    pins_path = tmp_path / "images.json"
    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(d, "PINS", pins_path)
    monkeypatch.setattr(d, "RECEIPT", receipt_path)
    original = {
        "services": {
            s: {"image": "old-" + component}
            for component, services in d.SERVICES.items()
            for s in services
        }
    }
    d.atomic_json(pins_path, original)
    sha = "a" * 40
    image = "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "b" * 64

    class Host(d.Host):
        def __init__(self) -> None:
            self.targets = targets
            self.calls: list[tuple[str, ...]] = []
            self.checks = 0

        def current_sha(self, component: str, token: str) -> str:
            return sha

        def run(self, *args: str, stdin: str | None = None) -> str:
            self.calls.append(args)
            return sha if "inspect" in args else ""

        def healthy(self, component: str) -> None:
            self.checks += 1
            if self.checks == 1:
                raise RuntimeError("second instance unhealthy")

    host = Host()
    with pytest.raises(RuntimeError, match="second instance"):
        d.deploy(
            "backend",
            {
                "sha": sha,
                "image": image,
                "registry_username": "actor",
                "registry_token": "token",
            },
            host,
        )
    assert json.loads(pins_path.read_text()) == original
    assert not receipt_path.exists()
    for target in targets:
        calls = [call for call in host.calls if call[0] == target.compose]
        assert sum("migrate" in call for call in calls) == 1
        ups = [call for call in calls if "up" in call]
        assert len(ups) == 2
        assert all(("bot" in call) == target.bot for call in ups)
        assert all("api" in call and "web" not in call for call in ups)


def test_web_update_starts_both_webs_without_touching_bots_or_databases() -> None:
    class Host(d.Host):
        targets = d.parse_targets(configured_targets())

        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.checked = False

        def run(self, *args: str, stdin: str | None = None) -> str:
            self.calls.append(args)
            return ""

        def healthy(self, component: str) -> None:
            assert component == "web"
            self.checked = True

    host = Host()
    d.start(host, "web", migrate=True)
    assert len(host.calls) == 2
    assert {call[0] for call in host.calls} == {target.compose for target in host.targets}
    assert all(call[-1] == "web" for call in host.calls)
    assert all("migrate" not in call and "bot" not in call for call in host.calls)
    assert host.checked


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../evil"},
        {"bot": "false"},
        {"api_port": True},
        {"web_port": 22},
        {"api_port": 13001},
        {"compose": "/bin/sh"},
    ],
)
def test_root_target_configuration_rejects_unsafe_or_ambiguous_values(
    change: dict,
) -> None:
    targets = configured_targets()
    targets[0].update(change)
    with pytest.raises(ValueError):
        d.parse_targets(targets)


def test_root_target_configuration_rejects_duplicate_instances_and_ports() -> None:
    targets = configured_targets()
    targets[1]["api_port"] = targets[0]["api_port"]
    with pytest.raises(ValueError):
        d.parse_targets(targets)
    with pytest.raises(ValueError):
        d.parse_targets([configured_targets()[0]] * 2)
    with pytest.raises(ValueError):
        d.parse_targets([])


def test_missing_config_preserves_legacy_nsu_and_unsafe_file_fails_closed(
    tmp_path: Path,
) -> None:
    assert d.load_targets(tmp_path / "missing") == (d.DEFAULT_TARGET,)
    config = tmp_path / "targets.json"
    config.write_text(json.dumps(configured_targets()))
    config.chmod(0o666)
    with pytest.raises(ValueError):
        d.load_targets(config)
    link = tmp_path / "targets-link.json"
    link.symlink_to(config)
    with pytest.raises(ValueError):
        d.load_targets(link)


def test_health_requires_both_instances_to_answer_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    paths: list[tuple[int, str]] = []

    class Handler(BaseHTTPRequestHandler):
        status = 200

        def do_GET(self) -> None:
            paths.append((self.status, self.path))
            self.send_response(self.status)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format: str, *args: object) -> None:
            pass

    class Unavailable(Handler):
        status = 503

    servers = [
        ThreadingHTTPServer(("127.0.0.1", 0), Handler),
        ThreadingHTTPServer(("127.0.0.1", 0), Unavailable),
    ]
    threads = [
        Thread(target=lambda s=s: s.serve_forever(poll_interval=0.005), daemon=True)
        for s in servers
    ]
    for thread in threads:
        thread.start()
    monkeypatch.setattr(d, "TARGETS_FILE", tmp_path / "missing")
    monkeypatch.setattr(d, "HEALTH_TIMEOUT", 0.05)
    monkeypatch.setattr(d, "HEALTH_INTERVAL", 0.005)
    host = d.Host(str(tmp_path))
    host.targets = (
        d.Target("nsu", servers[0].server_port, 13001, False),
        d.Target("nstu", servers[1].server_port, 13002, False),
    )
    try:
        with pytest.raises(RuntimeError, match="Health deadline"):
            host.healthy("backend")
        assert (200, "/api/v1/university/profile") in paths
        assert (503, "/api/v1/university/profile") in paths
        Unavailable.status = 200
        host.healthy("backend")
    finally:
        for server, thread in zip(servers, threads, strict=True):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


@pytest.mark.parametrize("change", ["add_instance", "port", "bot", "legacy_receipt"])
def test_same_image_reconciles_changed_target_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    monkeypatch.setattr(d, "PINS", tmp_path / "images.json")
    monkeypatch.setattr(d, "RECEIPT", tmp_path / "receipts.json")
    d.atomic_json(
        d.PINS,
        {
            "services": {
                s: {"image": "old-" + component}
                for component, services in d.SERVICES.items()
                for s in services
            }
        },
    )
    sha = "a" * 40
    data = {
        "sha": sha,
        "image": "ghcr.io/nsu-abitur-bot/abitur-bot@sha256:" + "b" * 64,
        "registry_username": "actor",
        "registry_token": "token",
    }

    class Host(d.Host):
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def current_sha(self, component: str, token: str) -> str:
            return sha

        def run(self, *args: str, stdin: str | None = None) -> str:
            self.calls.append(args)
            return sha if "inspect" in args else ""

        def healthy(self, component: str) -> None:
            pass

    d.deploy("backend", data, Host())
    retry = Host()
    if change == "add_instance":
        retry.targets = d.parse_targets(configured_targets())
    elif change == "port":
        retry.targets = (d.Target("nsu", 18003, 13001, True),)
    elif change == "bot":
        retry.targets = (d.Target("nsu", 18001, 13001, False),)
    else:
        receipt = json.loads(d.RECEIPT.read_text())
        receipt["backend"].pop("targets", None)
        d.atomic_json(d.RECEIPT, receipt)
    assert d.deploy("backend", data, retry) == "Deployment healthy"
    for target in retry.targets:
        assert any(call[0] == target.compose and "up" in call for call in retry.calls)
    receipt = json.loads(d.RECEIPT.read_text())["backend"]
    assert receipt["instances"] == [target.name for target in retry.targets]
    assert len(receipt["targets"]) == len(retry.targets)
    assert receipt["previous_image"] == "old-backend"
