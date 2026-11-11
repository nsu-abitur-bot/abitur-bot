"""Safety boundaries for the opt-in, paid CRAG comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.benchmark_crag_reuse import run


@pytest.mark.asyncio
async def test_benchmark_rejects_non_dedicated_database_before_initialization(
    monkeypatch,
):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/abitur")
    with pytest.raises(ValueError, match="Dedicated crag219_.*required"):
        await run(argparse.Namespace(), {})


def test_existing_report_prevents_paid_run_without_printing_private_input(tmp_path):
    report = tmp_path / "existing.json"
    report.write_text("original")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.benchmark_crag_reuse",
            "--execute",
            "--baseline",
            str(tmp_path / "not-executed.py"),
            "--output",
            str(report),
        ],
        input='{"private": "secret-marker-219"}',
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "error_type": "FileExistsError",
    }
    assert report.read_text() == "original"
    assert "secret-marker-219" not in result.stdout + result.stderr
