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
        for name in ("permissions", "migrate", "api"):
            assert services[name]["image"] == "ghcr.io/nsu-abitur-bot/abitur-bot:main"
            assert "build" not in services[name]
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


def test_publish_image_workflow_concurrency_and_tags() -> None:
    import yaml

    workflow_path = ROOT / ".github/workflows/publish-image.yml"
    assert workflow_path.exists()
    data = yaml.safe_load(workflow_path.read_text())
    assert "concurrency" in data
    assert data["concurrency"]["cancel-in-progress"] is True
    assert "github.ref" in str(data["concurrency"]["group"])
    tags = data["jobs"]["publish"]["steps"][-1]["with"]["tags"]
    assert "ghcr.io/${{ github.repository }}:main" in tags
    assert "ghcr.io/${{ github.repository }}:sha-${{ github.sha }}" in tags


def test_demo_instances_doc_rebuilds_web_on_update() -> None:
    doc_path = ROOT / "docs/demo-instances.md"
    content = doc_path.read_text()
    assert (
        "docker compose --env-file /tmp/abitur-demo/a.env -f deploy/demo/compose.yaml "
        "up -d --build --force-recreate"
    ) in content
