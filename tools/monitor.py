"""External monitor. Run once per minute with a systemd timer."""

import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def transition(state: dict, reasons: list[str], now: float) -> tuple[dict, str | None]:
    state = dict(state)
    state["failures"] = state.get("failures", 0) + 1 if reasons else 0
    message = None
    if reasons and state["failures"] >= 2:
        if not state.get("active") or now - state.get("sent_at", 0) >= 1800:
            message = "ALERT: " + "; ".join(reasons)
            state.update(active=True, sent_at=now)
    elif not reasons and state.get("active"):
        message = "RESOLVED: проверки прошли; проверьте ответ бота вручную"
        state.update(active=False, sent_at=now)
    return state, message


def check(target: dict) -> list[str]:
    reasons = []
    try:
        command = target.get("probe_command") or [
            "ssh",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=yes",
            "-oConnectTimeout=10",
            target["ssh"],
            "probe",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        data = json.loads(result.stdout)
        if data["failed"]:
            reasons.append("контейнеры: " + ", ".join(data["failed"]))
        current, previous = data["errors"], data["previous_errors"]
        if current >= 20 or (current >= 5 and current >= 2 * max(previous, 1)):
            reasons.append(f"ERROR/CRITICAL за 5 мин: {current}, ранее: {previous}")
    except Exception:
        reasons.append("сервер/SSH/Docker или сбор метрик недоступен")
    for url in target["urls"]:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    raise ValueError("HTTP status")
        except Exception:
            reasons.append("HTTP недоступен: " + url)
    return reasons


def send(token: str, chat_id: str, message: str) -> None:
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if not json.load(response).get("ok"):
            raise RuntimeError("delivery failed")


def run(config: dict, path: Path, test: bool = False) -> None:
    token = Path(config["token_file"]).read_text().strip()
    if test:
        for chat_id in config["chat_ids"]:
            send(token, chat_id, "TEST: мониторинг Abitur; " + config["runbook_url"])
        return
    state = json.loads(path.read_text()) if path.exists() else {}
    delivery_failed = False
    for target in config["targets"]:
        name = target["name"]
        reasons = check(target)
        for chat_id in config["chat_ids"]:
            key = f"{name}:{chat_id}"
            before = state.get(key, {})
            after, message = transition(before, reasons, time.time())
            if message:
                try:
                    send(token, chat_id, f"{name}: {message}\n{config['runbook_url']}")
                except Exception:
                    # Retry independently for each recipient after delivery failure.
                    delivery_failed = True
                    continue
            state[key] = after
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    os.replace(temporary, path)
    if delivery_failed:
        raise RuntimeError("delivery failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--state", type=Path, default=Path("/var/lib/abitur-monitor/state.json")
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    try:
        run(json.loads(args.config.read_text()), args.state, args.test)
    except Exception:
        raise SystemExit(
            "monitor failed; check config, access and Telegram delivery"
        ) from None


if __name__ == "__main__":
    main()
