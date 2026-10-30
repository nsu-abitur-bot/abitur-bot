"""Configuration checks that do not need a Docker daemon or provider keys."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_two_demo_configs_have_distinct_storage_and_secrets(tmp_path: Path) -> None:
    configs = []
    environments = []
    for name, api, web in (("demo-a", 18001, 13001), ("demo-b", 18002, 13002)):
        env = tmp_path / f"{name}.env"
        args = [
            sys.executable,
            str(ROOT / "deploy/demo/create_env.py"),
            str(env),
            name,
            "--api-port",
            str(api),
            "--web-port",
            str(web),
        ]
        subprocess.run(args, check=True)
        original = env.read_text()
        assert env.stat().st_mode & 0o777 == 0o600
        assert subprocess.run(args, capture_output=True).returncode != 0
        assert env.read_text() == original
        environments.append(dict(line.split("=", 1) for line in original.splitlines()))
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(env),
                "-f",
                str(ROOT / "deploy/demo/compose.yaml"),
                "config",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        configs.append(json.loads(result.stdout))
    a, b = configs
    assert set(v["name"] for v in a["volumes"].values()).isdisjoint(
        v["name"] for v in b["volumes"].values()
    )
    assert a["networks"]["default"]["name"] != b["networks"]["default"]["name"]
    for key in ("SECRET_KEY", "SETTINGS_ENCRYPTION_KEY", "DB_PASSWORD", "REDIS_PASSWORD"):
        assert environments[0][key] != environments[1][key]
    for config, environment in zip(configs, environments, strict=True):
        services = config["services"]
        assert "bot" not in services and "scheduler" not in services
        assert "ports" not in services["postgres"]
        assert "ports" not in services["redis"]
        assert (
            services["web"]["environment"]["NUXT_PUBLIC_API_BASE_URL"]
            == (environment["API_ORIGIN"])
        )
        assert not any(
            volume["type"] == "bind"
            for service in services.values()
            for volume in service.get("volumes", [])
        )
