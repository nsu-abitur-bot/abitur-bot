"""Read-only Compose probe; stdout contains aggregates, never application logs."""

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime, timedelta

ERROR = re.compile(r" - (?:ERROR|CRITICAL) - |^ERROR:|^CRITICAL:")


def error_counts(logs: str, now: datetime) -> tuple[int, int]:
    current = previous = 0
    for line in logs.splitlines():
        stamp, _, message = line.partition(" ")
        if not ERROR.search(message):
            continue
        try:
            age = (
                now - datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            ).total_seconds()
        except ValueError:
            continue
        if 0 <= age < 300:
            current += 1
        elif 300 <= age < 600:
            previous += 1
    return current, previous


def probe(wrapper: str, services: list[str]) -> dict:
    def compose(*args: str) -> str:
        return subprocess.run(
            [wrapper, *args], check=True, capture_output=True, text=True, timeout=25
        ).stdout

    raw = compose("ps", "--all", "--format", "json")
    rows = (
        json.loads(raw)
        if raw.lstrip().startswith("[")
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )
    failed = []
    for service in services:
        containers = [row for row in rows if row["Service"] == service]
        if not containers or any(
            row["State"] != "running" or row.get("Health", "") not in ("", "healthy")
            for row in containers
        ):
            failed.append(service)
    now = datetime.now(UTC)
    # Docker timestamps avoid dependence on the application/server timezone.
    logs = compose(
        "logs",
        "--no-color",
        "--no-log-prefix",
        "--timestamps",
        "--since",
        (now - timedelta(minutes=10)).isoformat(),
        "--until",
        now.isoformat(),
        *services,
    )
    current, previous = error_counts(logs, now)
    return {"failed": failed, "errors": current, "previous_errors": previous}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrapper")
    parser.add_argument("services", nargs="+")
    args = parser.parse_args()
    try:
        result = probe(args.wrapper, args.services)
    except Exception:
        # subprocess output may contain secrets; the monitor only needs failure.
        raise SystemExit("probe failed") from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
