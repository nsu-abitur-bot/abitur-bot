"""Send credentials on SSH stdin, never in argv or shell expansions."""

import json
import os
import subprocess


def main() -> None:
    payload = {
        "image": os.environ["DEPLOY_IMAGE"],
        "sha": os.environ["DEPLOY_SHA"],
        "registry_username": os.environ["GITHUB_ACTOR"],
        "registry_token": os.environ["GH_TOKEN"],
    }
    subprocess.run(
        [
            "ssh",
            "-T",
            "-p",
            "22",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=40",
            "-i",
            os.path.expanduser("~/.ssh/deploy_key"),
            os.environ["DEPLOY_USER"] + "@" + os.environ["DEPLOY_HOST"],
            "deploy",
        ],
        input=json.dumps(payload),
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
