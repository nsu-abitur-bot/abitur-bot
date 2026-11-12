import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.admission_index_one import claim_reports, pending_resume


def test_import_does_not_disable_process_logging():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging; before=logging.root.manager.disable; "
            "import tools.admission_index_one; "
            "raise SystemExit(logging.root.manager.disable != before)",
        ],
        capture_output=True,
    )
    assert result.returncode == 0


def test_safety_checks_remain_active_under_python_optimization():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from tools.admission_index_one import require; require(False, 'blocked')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SafetyCheckFailed" in result.stderr


def test_run_marker_survives_and_prevents_retry(tmp_path):
    report, usage = tmp_path / "result.json", tmp_path / "usage.jsonl"
    claim_reports(report, usage)
    if os.name == "posix":
        assert report.stat().st_mode & 0o777 == 0o640
        assert usage.stat().st_mode & 0o777 == 0o640
    assert json.loads(report.read_text()) == {"status": "started"}
    with pytest.raises(RuntimeError):
        claim_reports(report, usage)
    assert json.loads(report.read_text()) == {"status": "started"}


def test_resume_rejects_a_changed_or_completed_source():
    item = {"text": "pinned source", "source_hash": "a" * 64, "graph_id": "abitur_kb"}
    record = dict(item, id="same-proposal", status="pending")
    assert pending_resume([record], item, "same-proposal") == "same-proposal"
    with pytest.raises(RuntimeError):
        pending_resume([dict(record, text="other source")], item, "same-proposal")
    with pytest.raises(RuntimeError):
        pending_resume([dict(record, status="approved")], item, "same-proposal")


def _linux_launcher_shell() -> str:
    if sys.platform != "linux":
        pytest.skip("Launcher lifecycle requires Linux; run these cases in Linux CI")
    tools = {name: shutil.which(name) for name in ("sh", "timeout", "flock")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        pytest.skip("Launcher lifecycle requires tools: " + ", ".join(missing))
    shell, timeout = tools["sh"], tools["timeout"]
    assert shell and timeout
    try:
        version = subprocess.run(
            [timeout, "--version"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"Launcher lifecycle requires GNU timeout ({type(exc).__name__})")
    if version.returncode != 0 or "GNU coreutils" not in version.stdout:
        pytest.skip("Launcher lifecycle requires GNU timeout from coreutils")
    return shell


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_launcher_guard_skips_non_linux(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(shutil, "which", lambda tool: f"/synthetic/bin/{tool}")
    with pytest.raises(pytest.skip.Exception, match="requires Linux"):
        _linux_launcher_shell()


@pytest.mark.parametrize("missing", ["sh", "timeout", "flock"])
def test_launcher_guard_skips_missing_tools(monkeypatch, missing):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        shutil, "which", lambda tool: None if tool == missing else f"/synthetic/{tool}"
    )
    with pytest.raises(pytest.skip.Exception, match=missing):
        _linux_launcher_shell()


@pytest.mark.parametrize("gnu", [False, True])
def test_launcher_guard_requires_gnu_timeout(monkeypatch, gnu):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda tool: f"/synthetic/{tool}")

    def version(command, **kwargs):
        assert command == ["/synthetic/timeout", "--version"]
        return subprocess.CompletedProcess(
            command, 0, "timeout (GNU coreutils) 9.4" if gnu else "BusyBox timeout", ""
        )

    monkeypatch.setattr(subprocess, "run", version)
    if gnu:
        assert _linux_launcher_shell() == "/synthetic/sh"
    else:
        with pytest.raises(pytest.skip.Exception, match="GNU timeout"):
            _linux_launcher_shell()


@pytest.mark.parametrize("failure", ["none", "start", "run", "stop", "existing"])
@pytest.mark.parametrize("launcher", ["run-one.sh", "run-remaining.sh"])
def test_launcher_stops_only_its_project_on_success_and_failure(
    tmp_path, failure, launcher
):
    shell = _linux_launcher_shell()
    root = Path(__file__).resolve().parents[2]
    source = (root / "deploy/evals/388" / launcher).read_text()
    # Test-only path substitutions keep this lifecycle test outside /home and sudo.
    source = source.replace("/home/mihail/abitur-evals/388-20261107", str(tmp_path))
    source = source.replace(
        "/home/mihail/abitur-evals/388-reports/with-kb", str(tmp_path / "reports")
    )
    script = tmp_path / "run-one.sh"
    script.write_text(source)
    commands = tmp_path / "commands"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        "#!/bin/sh\n"
        'test "$1" = -n && test "$2" = /usr/local/sbin/abitur-eval388 || exit 88\n'
        'test "$3" = with-kb || exit 89\n'
        'printf "%s\\n" "$4" >> "$COMMANDS"\n'
        'case "$4" in\n'
        ' status) printf "NAME IMAGE\\n";\n'
        ' if [ "$FAILURE" = existing ]; then echo "old api"; fi;;\n'
        ' start|run|stop) if [ "$FAILURE" = "$4" ]; then exit 23; fi;;\n'
        " *) exit 90;;\nesac\n"
    )
    sudo.chmod(0o700)
    environment = dict(
        os.environ,
        PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
        COMMANDS=str(commands),
        FAILURE=failure,
    )
    result = subprocess.run(
        [shell, str(script), "--execute"],
        env=environment,
        capture_output=True,
        text=True,
        input="{}",
    )
    steps = commands.read_text().splitlines()
    if failure == "existing":
        assert steps == ["status"] and result.returncode != 0
    else:
        assert steps[-1] == "stop"
        assert steps == (
            ["status", "start", "stop"]
            if failure == "start"
            else ["status", "start", "run", "stop"]
        )
        assert (result.returncode == 0) == (failure == "none")
