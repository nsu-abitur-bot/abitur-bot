import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.admission_index_one import claim_reports


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
    assert json.loads(report.read_text()) == {"status": "started"}
    with pytest.raises(RuntimeError):
        claim_reports(report, usage)
    assert json.loads(report.read_text()) == {"status": "started"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifecycle verified in Linux CI")
@pytest.mark.parametrize("failure", ["none", "start", "run", "stop", "existing"])
def test_launcher_stops_only_its_project_on_success_and_failure(tmp_path, failure):
    shell = shutil.which("sh")
    assert shell
    root = Path(__file__).resolve().parents[2]
    source = (root / "deploy/evals/388/run-one.sh").read_text()
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
