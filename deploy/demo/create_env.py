"""Create infrastructure-only configuration; never overwrite an existing secret."""

import argparse
import base64
import os
import re
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("name")
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--web-port", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,40}", args.name):
        parser.error("name must match [a-z][a-z0-9_-]{2,40}")
    if not all(1024 <= port <= 65535 for port in (args.api_port, args.web_port)):
        parser.error("ports must be between 1024 and 65535")
    if args.api_port == args.web_port:
        parser.error("API and UI ports must differ")
    values = {
        "INSTANCE_NAME": args.name,
        "BIND_ADDRESS": "127.0.0.1",
        "API_PORT": str(args.api_port),
        "WEB_PORT": str(args.web_port),
        "API_ORIGIN": f"http://localhost:{args.api_port}",
        "WEB_ORIGIN": f"http://localhost:{args.web_port}",
        "DB_PASSWORD": secrets.token_hex(24),
        "REDIS_PASSWORD": secrets.token_hex(24),
        "SECRET_KEY": secrets.token_hex(32),
        "SETTINGS_ENCRYPTION_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(),
        "WEB_SESSION_PASSWORD": secrets.token_hex(32),
    }
    fd = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write("".join(f"{key}={value}\n" for key, value in values.items()))


if __name__ == "__main__":
    main()
