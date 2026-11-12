"""Exercise real Docker probes and Telegram delivery on an isolated container."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from monitor import run  # noqa: E402


def main() -> None:
    base = Path(__file__).resolve().parent
    config = json.loads(Path("/etc/abitur-monitor/config.json").read_text())
    command = ["docker", "compose", "-f", str(base / "drill.yaml")]
    with tempfile.TemporaryDirectory(prefix="abitur-monitor-drill-") as directory:
        root = Path(directory)
        wrapper = root / "compose"
        wrapper.write_text(
            f'#!/bin/sh\nexec docker compose -f "{base}/drill.yaml" "$@"\n'
        )
        wrapper.chmod(0o700)
        config["targets"] = [
            {
                "name": "TEST isolated monitoring drill",
                "probe_command": [
                    "/usr/local/bin/uv",
                    "run",
                    "--no-project",
                    "--python",
                    "/usr/bin/python3",
                    str(base.parents[1] / "tools/monitor_probe.py"),
                    str(wrapper),
                    "sample",
                ],
                "urls": [],
            }
        ]
        state = root / "state.json"

        def compose(*args: str, errors: int = 0) -> None:
            subprocess.run(
                command + list(args),
                check=True,
                capture_output=True,
                env={**os.environ, "ERROR_COUNT": str(errors)},
                timeout=30,
            )

        try:
            compose("up", "-d", "--pull", "never", errors=20)
            # Wait for the fixture's actual ERROR output, not an arbitrary delay.
            subprocess.run(
                command + ["exec", "-T", "sample", "true"],
                check=True,
                capture_output=True,
                timeout=15,
            )
            run(config, state)
            run(config, state)
            assert all(s.get("active") for s in json.loads(state.read_text()).values())
            print("error alert delivered")
            compose("up", "-d", "--force-recreate", "--pull", "never")
            run(config, state)
            assert all(
                not s.get("active") for s in json.loads(state.read_text()).values()
            )
            print("error recovery delivered")
            compose("stop")
            run(config, state)
            run(config, state)
            assert all(s.get("active") for s in json.loads(state.read_text()).values())
            print("stopped container alert delivered")
            compose("start")
            run(config, state)
            assert all(
                not s.get("active") for s in json.loads(state.read_text()).values()
            )
            print("container recovery delivered")
        finally:
            compose("down")


if __name__ == "__main__":
    main()
