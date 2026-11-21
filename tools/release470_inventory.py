"""Read-only Linux metadata for #470, without application imports or secret files."""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENTRYPOINTS = {
    "eval388": "/usr/local/sbin/abitur-eval388",
    "eval470": "/usr/local/sbin/abitur-eval470",
    "nsu_compose": "/usr/local/bin/abitur-nsu-compose",
    "nsu_probe": "/usr/local/sbin/abitur-nsu-probe",
}


def path_metadata(value: str) -> dict[str, Any]:
    """Return metadata only. Do not open wrappers or their configuration."""
    try:
        info = Path(value).stat()
    except FileNotFoundError:
        return {"exists": False}
    except PermissionError:
        return {"exists": None, "metadata_access": False}
    return {
        "exists": True,
        "kind": "directory" if stat.S_ISDIR(info.st_mode) else "file",
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "owner_uid": info.st_uid,
        "readable": os.access(value, os.R_OK),
        "executable": os.access(value, os.X_OK),
    }


def inventory() -> dict[str, Any]:
    """Do not start services, invoke wrappers, read logs or inspect environments."""
    if sys.platform != "linux":
        raise RuntimeError("Run this read-only inventory on the Linux host")
    meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
    memory = {
        name: int(value)
        for name, value in re.findall(r"^(\w+):\s+(\d+) kB$", meminfo, re.M)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    }
    disk = shutil.disk_usage("/")
    sudo = subprocess.run(
        ["sudo", "-n", "-l"], capture_output=True, text=True, timeout=10, check=False
    )
    # Only whitelist booleans leave this process. Never include raw sudo output.
    listed = {
        name: bool(re.search(re.escape(path) + r"(?:\s|,|$)", sudo.stdout))
        if sudo.returncode == 0
        else None
        for name, path in ENTRYPOINTS.items()
    }
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "unprivileged host metadata; no app, DB, logs or model calls",
        "memory_kib": memory,
        "root_disk_bytes": {"total": disk.total, "free": disk.free},
        "sudo": {"query_exit_code": sudo.returncode, "explicit_paths_listed": listed},
        "entrypoints": {name: path_metadata(path) for name, path in ENTRYPOINTS.items()},
        "nsu_runtime_configuration_checked": False,
        "application_data_read": False,
        "services_started": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(inventory(), indent=2, sort_keys=True))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        # Exception messages and child output might contain host-specific data.
        print(json.dumps({"inventory_error": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
