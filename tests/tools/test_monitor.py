import json
from datetime import UTC, datetime, timedelta

import pytest

from tools import monitor
from tools.monitor_probe import error_counts, probe


def test_alert_repeat_recovery():
    state, message = monitor.transition({}, ["api"], 100)
    assert message is None
    state, message = monitor.transition(state, ["api"], 160)
    assert message == "ALERT: api"
    state, message = monitor.transition(state, ["api"], 220)
    assert message is None
    state, message = monitor.transition(state, ["api"], 1960)
    assert message == "ALERT: api"
    state, message = monitor.transition(state, [], 2020)
    assert message is not None and message.startswith("RESOLVED")
    assert monitor.transition(state, [], 2080)[1] is None


def test_single_failure_does_not_alert():
    state, _ = monitor.transition({}, ["api"], 100)
    assert monitor.transition(state, [], 160)[1] is None


def test_error_windows_ignore_tracebacks_and_old_entries():
    now = datetime.now(UTC)
    logs = "\n".join(
        f"{(now - timedelta(seconds=age)).isoformat()} {message}"
        for age, message in [
            (0, "2026-11-11 - pipeline - ERROR - failed"),
            (299, "ERROR:    ASGI failed"),
            (300, "2026-11-11 - bot - CRITICAL - failed"),
            (600, "ERROR: old"),
            (10, '    raise ValueError("ERROR")'),
            (10, "2026-11-11 - bot - INFO - hello"),
        ]
    )
    assert error_counts(logs, now) == (2, 1)


@pytest.mark.parametrize(
    "errors,previous,alert",
    [(4, 0, False), (5, 0, True), (10, 6, False), (12, 6, True), (20, 30, True)],
)
def test_error_thresholds(monkeypatch, errors, previous, alert):
    class Result:
        stdout = json.dumps({"failed": [], "errors": errors, "previous_errors": previous})

    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **kw: Result())
    assert bool(monitor.check({"ssh": "test", "urls": []})) == alert


def test_unreachable_host(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("secret stderr")

    monkeypatch.setattr(monitor.subprocess, "run", fail)
    reasons = monitor.check({"ssh": "test", "urls": []})
    assert len(reasons) == 1
    assert "secret" not in str(reasons)


def test_failed_delivery_retried_with_persistent_state(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("test")
    state = tmp_path / "state.json"
    config = {
        "token_file": str(token),
        "chat_ids": ["test"],
        "runbook_url": "runbook",
        "targets": [{"name": "nsu"}],
    }
    monkeypatch.setattr(monitor, "check", lambda target: ["api"])
    monitor.run(config, state)

    def fail(*args):
        raise OSError("offline")

    monkeypatch.setattr(monitor, "send", fail)
    with pytest.raises(RuntimeError):
        monitor.run(config, state)
    assert not json.loads(state.read_text())["nsu:test"].get("active")
    messages = []
    monkeypatch.setattr(monitor, "send", lambda *args: messages.append(args[-1]))
    monitor.run(config, state)
    monitor.run(config, state)
    assert len(messages) == 1
    assert "ALERT" in messages[0]


@pytest.mark.parametrize("array", [True, False])
def test_real_probe_subprocess_missing_stopped_unhealthy(tmp_path, array):
    rows = [
        {"Service": "api", "State": "running", "Health": "unhealthy"},
        {"Service": "redis", "State": "exited", "Health": ""},
        {"Service": "postgres", "State": "running", "Health": "healthy"},
    ]
    payload = json.dumps(rows) if array else "\n".join(map(json.dumps, rows))
    fixture = tmp_path / "ps.json"
    fixture.write_text(payload)
    wrapper = tmp_path / "compose"
    wrapper.write_text(f'#!/bin/sh\nif [ "$1" = ps ]; then cat "{fixture}"; fi\n')
    wrapper.chmod(0o700)
    result = probe(str(wrapper), ["api", "redis", "postgres", "bot"])
    assert result == {
        "failed": ["api", "redis", "bot"],
        "errors": 0,
        "previous_errors": 0,
    }


def test_recipients_retry_independently(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("test")
    path = tmp_path / "state.json"
    config = {
        "token_file": str(token),
        "chat_ids": ["first", "second"],
        "runbook_url": "runbook",
        "targets": [{"name": "nsu"}],
    }
    monkeypatch.setattr(monitor, "check", lambda target: ["api"])
    sent = []

    def deliver(token, chat, message):
        if chat == "second":
            raise OSError("offline")
        sent.append(chat)

    monkeypatch.setattr(monitor, "send", deliver)
    monitor.run(config, path)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            monitor.run(config, path)
    assert sent == ["first"]
    monkeypatch.setattr(monitor, "send", lambda token, chat, message: sent.append(chat))
    monitor.run(config, path)
    assert sent == ["first", "second"]
