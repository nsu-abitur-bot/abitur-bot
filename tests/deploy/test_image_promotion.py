"""Exercise the promotion boundary without GitHub, a registry or Docker daemon."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_delayed_old_run_cannot_replace_current_main(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    head = tmp_path / "head"
    published = tmp_path / "published"
    for command, body in {
        "gh": (
            'test "$*" = "api repos/owner/repo/git/ref/heads/main --jq .object.sha"\n'
            'cat "$REMOTE_HEAD"'
        ),
        "docker": (
            'printf "%s\\n" "$*" >> "$PUBLISHED"\n'
            'if [ -n "${NEXT_HEAD:-}" ]; then printf "%s" "$NEXT_HEAD" '
            '> "$REMOTE_HEAD"; fi'
        ),
    }.items():
        executable = bin_dir / command
        executable.write_text(f"#!/bin/sh\nset -eu\n{body}\n")
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_HEAD": str(head),
        "PUBLISHED": str(published),
        "GITHUB_REPOSITORY": "owner/repo",
        "IMAGE_DIGEST": "sha256:" + "b" * 64,
    }

    def promote(commit: str, next_head: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(ROOT / "tools/promote_backend_image.sh")],
            env={**env, "TESTED_SHA": commit, "NEXT_HEAD": next_head},
            text=True,
            capture_output=True,
            check=True,
        )

    a, b = "a" * 40, "b" * 40
    # A push during A's registry write cannot let A overwrite B: B still waits
    # for the job lock and publishes afterward. No atomic Git/GHCR transaction.
    head.write_text(a)
    promote(a, next_head=b)
    assert head.read_text() == b
    published.unlink()
    promote(b)  # B acquires the publisher lock before delayed A.
    first_publication = published.read_text()
    assert first_publication == (
        "buildx imagetools create --tag ghcr.io/owner/repo:main "
        f"ghcr.io/owner/repo@{env['IMAGE_DIGEST']}\n"
    )
    assert "Skipping" in promote(a).stdout
    assert published.read_text() == first_publication
    assert "Skipping" in promote(a).stdout  # Manual rerun of the old checkout.
    assert published.read_text() == first_publication
    head.write_text("c" * 40)  # A newer push arrives before B gets its lock.
    assert "Skipping" in promote(b).stdout
    assert published.read_text() == first_publication


def test_remote_lookup_failure_does_not_publish(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\nexit 1\n")
    gh.chmod(0o755)
    published = tmp_path / "published"
    docker = bin_dir / "docker"
    docker.write_text(f"#!/bin/sh\ntouch '{published}'\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "tools/promote_backend_image.sh")],
        env={
            **os.environ,
            "PATH": str(bin_dir) + ":" + os.environ["PATH"],
            "GITHUB_REPOSITORY": "owner/repo",
            "TESTED_SHA": "a" * 40,
            "IMAGE_DIGEST": "sha256:" + "a" * 64,
        },
        capture_output=True,
    )
    assert result.returncode != 0
    assert not published.exists()
