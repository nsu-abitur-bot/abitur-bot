"""Measure imports and graph loading on a private copy, without model requests."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

FILES = (
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_full_entities.json",
    "kv_store_full_relations.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
    "kv_store_llm_response_cache.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "graph_chunk_entity_relation.graphml",
)


def require(value: object) -> None:
    if not value:
        raise RuntimeError("Memory probe safety check failed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_copy(source: Path, target_base: Path) -> dict[str, str]:
    require(source.resolve() == source and source.is_dir())
    selected = []
    for name in FILES:
        path = source / name
        require(not path.is_symlink())
        if name == "kv_store_llm_response_cache.json" and not path.exists():
            continue
        require(path.is_file())
        selected.append(path)
    require(sum(path.stat().st_size for path in selected) <= 512 * 1024**2)
    require(not target_base.is_symlink())
    target_base.mkdir()
    target = target_base / "abitur_kb"
    target.mkdir()
    hashes = {}
    for path in selected:
        hashes[path.name] = sha256(path)
        shutil.copyfile(path, target / path.name)
        require(sha256(target / path.name) == hashes[path.name])
    return hashes


def snapshot() -> dict:
    import resource

    cgroup = Path("/sys/fs/cgroup")
    values: dict = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current"):
        value = (cgroup / name).read_text().strip()
        values[name] = int(value)
    values["memory.events"] = {
        key: int(value)
        for key, value in map(
            lambda line: line.split(), (cgroup / "memory.events").read_text().splitlines()
        )
        if key in {"oom", "oom_kill", "max"}
    }
    values["host_available_kib"] = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    values["process_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return values


class NetworkGuard:
    def __init__(self, addresses: set[tuple[str, int]]) -> None:
        self.addresses = addresses
        self.blocked = 0

    def audit(self, event: str, args: tuple) -> None:
        if event == "socket.getaddrinfo":
            if args[0] not in {"postgres", "redis"} | {a[0] for a in self.addresses}:
                self.blocked += 1
                raise RuntimeError("External DNS is disabled for memory probe")
        elif event == "socket.connect":
            address = args[1]
            if not isinstance(address, tuple) or address[:2] not in self.addresses:
                self.blocked += 1
                raise RuntimeError("External connections are disabled for memory probe")


class Recorder:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.finished = threading.Event()
        self.data: dict = {
            "run_id": run_id,
            "status": "started",
            "phase": "baseline",
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "phases": [],
            "min_host_available_kib": 2**63,
            "max_sampled_swap_bytes": 0,
            "generation_calls": 0,
            "embedding_calls": 0,
            "question_calls": 0,
            "api_server_started": False,
        }
        with path.open("x", encoding="utf-8") as output:
            json.dump(self.data, output)

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def record(self, phase: str | None = None) -> None:
        value = snapshot()
        with self.lock:
            if phase:
                self.data["phase"] = phase
                self.data["phases"].append({"phase": phase, **value})
            self.data["latest_memory"] = value
            self.data["min_host_available_kib"] = min(
                self.data["min_host_available_kib"], value["host_available_kib"]
            )
            self.data["max_sampled_swap_bytes"] = max(
                self.data["max_sampled_swap_bytes"], value["memory.swap.current"]
            )
            if value["host_available_kib"] < 300 * 1024:
                self.data["status"] = "aborted_low_host_memory"
                self.save()
                os._exit(75)
            self.save()

    def observe(self) -> None:
        while not self.finished.wait(0.2):
            try:
                self.record()
            except Exception:
                # An unobserved resource probe must not keep running.
                os._exit(76)


async def warm_graph(recorder: Recorder, run_id: str) -> None:
    from bootstrap.runtime_settings import refresh_settings
    from config.instance import get_setting
    from db.postgres.db import engine

    memory = None
    try:
        require(engine.url.host == "postgres" and engine.url.database == "eval388")
        await refresh_settings()
        require(get_setting("LIGHTRAG_STORAGE", "json") == "json")
        recorder.record("runtime_settings_loaded")
        from rag.graph_memory import GraphMemory

        recorder.record("application_imported")
        source = Path("/app/data/lightrag/abitur_kb")
        target_base = Path("/app/data") / f".memory-probe-{run_id}"
        target = target_base / "abitur_kb"
        hashes = prepare_copy(source, target_base)
        require(len(json.loads((target / "kv_store_doc_status.json").read_text())) == 9)
        recorder.record("private_copy_prepared")

        # Only this manager's workspace changes. Runtime provider settings stay intact.
        def setting(name: str, default: str = "") -> str:
            return (
                str(target_base)
                if name == "LIGHTRAG_WORKSPACE_BASE"
                else get_setting(name, default)
            )

        with patch("rag.graph_memory.get_setting", side_effect=setting):
            memory = GraphMemory()
            require(Path(memory.workspace_path) == target_base)
            async with memory._use_graph("abitur_kb"):
                recorder.record("graph_storages_loaded")
                await asyncio.sleep(2)
            await memory.cleanup()
            memory = None
        require(all(sha256(source / name) == digest for name, digest in hashes.items()))
        with recorder.lock:
            recorder.data["source_hashes_unchanged"] = True
            recorder.data["source_files"] = len(hashes)
            recorder.data["private_copy"] = str(target_base)
        recorder.record("graph_closed")
    finally:
        if memory is not None:
            await memory.cleanup()
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute)
    os.umask(0o027)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    run_id = os.environ.get("EVAL388_RUN_ID", "")
    require(re.fullmatch(r"[0-9a-f]{32}", run_id))
    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text()
    )
    require(os.getuid() == 1000)
    allowed = set()
    for host, port in (("postgres", 5432), ("redis", 6379)):
        allowed.update((str(row[4][0]), port) for row in socket.getaddrinfo(host, port))
    guard = NetworkGuard(allowed)
    sys.addaudithook(guard.audit)
    recorder = Recorder(Path(f"/reports/cli-memory-{run_id}.json"), run_id)
    recorder.record("baseline")
    observer = threading.Thread(target=recorder.observe, daemon=True)
    observer.start()
    code = 1
    try:
        asyncio.run(warm_graph(recorder, run_id))
        require(guard.blocked == 0)
        require(recorder.data["latest_memory"]["memory.events"]["oom_kill"] == 0)
        recorder.data["status"] = "completed"
        code = 0
    except Exception as exc:
        recorder.data["status"] = "failed"
        recorder.data["error_type"] = type(exc).__name__
    finally:
        recorder.finished.set()
        observer.join(timeout=2)
        recorder.data["blocked_network_attempts"] = guard.blocked
        recorder.record("finished")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
