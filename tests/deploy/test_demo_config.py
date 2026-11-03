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
    assert "concurrency" not in data
    publish = data["jobs"]["publish"]
    assert publish["outputs"]["digest"] == "${{ steps.build.outputs.digest }}"
    tags = publish["steps"][-1]["with"]["tags"]
    assert (
        tags == "ghcr.io/${{ github.repository }}:sha-"
        "${{ github.event.workflow_run.head_sha }}"
    )
    guard = publish["if"]
    for condition in (
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
        "github.event.workflow_run.conclusion == 'success'",
    ):
        assert condition in guard
    for job in data["jobs"].values():
        assert job["steps"][0]["with"]["ref"] == (
            "${{ github.event.workflow_run.head_sha }}"
        )
    promote = data["jobs"]["promote"]
    assert promote["needs"] == "publish"
    assert promote["concurrency"] == {
        "group": "backend-image-main-promotion",
        "cancel-in-progress": False,
        "queue": "max",
    }
    step = promote["steps"][-1]
    assert step["run"] == "bash tools/promote_backend_image.sh"
    assert step["env"]["IMAGE_DIGEST"] == "${{ needs.publish.outputs.digest }}"


def test_demo_instances_doc_rebuilds_web_on_update() -> None:
    doc_path = ROOT / "docs/demo-instances.md"
    content = doc_path.read_text()
    assert (
        "docker compose --env-file /tmp/abitur-demo/a.env -f deploy/demo/compose.yaml "
        "up -d --build --force-recreate"
    ) in content


def test_runtime_shares_only_provider_coordination_files(tmp_path: Path) -> None:
    configs = []
    for name, api, web in (("demo-a", 18001, 13001), ("demo-b", 18002, 13002)):
        env = tmp_path / (name + ".env")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "deploy/demo/create_env.py"),
                str(env),
                name,
                "--api-port",
                str(api),
                "--web-port",
                str(web),
            ],
            check=True,
        )
        with env.open("a") as output:
            output.write(f"API_HOST=api.{name}.example\nWEB_HOST={name}.example\n")
        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(env),
                "-f",
                str(ROOT / "deploy/demo/compose.yaml"),
                "-f",
                str(ROOT / "deploy/demo/runtime.yaml"),
                "--profile",
                "telegram",
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
    shared = set(v["name"] for v in a["volumes"].values()) & set(
        v["name"] for v in b["volumes"].values()
    )
    assert shared == {"abitur-provider-locks"}
    for config in configs:
        for name in ("api", "bot", "migrate"):
            service = config["services"][name]
            assert (
                service["environment"]["GIGACHAT_LOCK_DIRECTORY"] == "/app/provider-locks"
            )
            assert any(
                v["source"] == "provider-locks" and v["target"] == "/app/provider-locks"
                for v in service["volumes"]
            )
            assert any(
                v["source"] == "files" and v["target"] == "/app/data"
                for v in service["volumes"]
            )
