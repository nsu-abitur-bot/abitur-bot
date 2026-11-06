"""Restore a disposable local PostgreSQL + persistent LightRAG fixture, no LLM API."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


async def graph_fixture(path: Path, mode: str) -> None:
    import numpy as np
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import Tokenizer, wrap_embedding_func_with_attrs

    class Words:
        def __init__(self) -> None:
            self.words: list[str] = []

        def encode(self, content: str) -> list[int]:
            result = []
            for word in content.split(" "):
                if word not in self.words:
                    self.words.append(word)
                result.append(self.words.index(word))
            return result

        def decode(self, tokens: list[int]) -> str:
            return " ".join(self.words[token] for token in tokens)

    @wrap_embedding_func_with_attrs(embedding_dim=8)
    async def embed(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 8))

    async def model(prompt: str, **kwargs) -> str:
        return ""

    rag = LightRAG(
        working_dir=str(path),
        workspace="release390",
        llm_model_func=model,
        embedding_func=embed,
        tokenizer=Tokenizer("words", Words()),
        chunk_token_size=80,
        chunk_overlap_token_size=10,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    try:
        if mode in {"write", "append"}:
            identifier = "checkpoint" if mode == "write" else "after-checkpoint"
            await rag.ainsert(
                "SYNTHETIC_RELEASE_390 disposable restore fixture. " * 25,
                ids=[identifier],
            )
            record = await rag.doc_status.get_by_id(identifier)
            if not record or record.get("status") != "processed":
                raise RuntimeError("Synthetic document not processed")
        else:
            if await rag.full_docs.get_by_id("checkpoint") is None:
                raise RuntimeError("Restored document missing")
            if await rag.full_docs.get_by_id("after-checkpoint") is not None:
                raise RuntimeError("Unexpected document outside checkpoint")
            context = await rag.aquery(
                "SYNTHETIC_RELEASE_390", QueryParam(mode="naive", only_need_context=True)
            )
            if "SYNTHETIC_RELEASE_390" not in str(context):
                raise RuntimeError("Restored vector retrieval failed")
    finally:
        await rag.finalize_storages()


def run_fixture(args: argparse.Namespace) -> None:
    if not args.pg_bin or not args.workdir or not args.port:
        raise ValueError("Explicit pg-bin, workdir and isolated port required")
    if args.port < 49152 or not args.user.startswith("abitur_390"):
        raise ValueError("Use the disposable high-port abitur_390 cluster")
    if args.workdir.exists():
        raise ValueError("Workdir must be new, no overwrites")
    args.workdir.mkdir(parents=True)
    root = args.workdir.resolve()
    suffix = uuid4().hex[:12]
    source = f"abitur_390_source_{suffix}"
    target = f"abitur_390_target_{suffix}"
    env = dict(os.environ)
    env.update(
        PGPASSFILE=str(root / "no-password-file"),
        PGSERVICEFILE=str(root / "no-service-file"),
        PGSSLMODE="disable",
        PGPASSWORD="test",
        PGHOST="127.0.0.1",
        PGPORT=str(args.port),
        PGUSER=args.user,
        PYTHON_DOTENV_DISABLED="1",
        UV_NO_ENV_FILE="1",
    )
    env.pop("PGSERVICE", None)

    def pg(program: str, *params: str) -> str:
        binary = args.pg_bin / (program + (".exe" if os.name == "nt" else ""))
        result = subprocess.run(
            [str(binary), *params],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    def sql(database: str, query: str) -> str:
        return pg(
            "psql", "-X", "-At", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", query
        )

    def graph(directory: Path, mode: str) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.release_restore_probe",
                "--graph",
                str(directory),
                "--mode",
                mode,
            ],
            env=env,
            check=True,
            capture_output=True,
        )

    pg("createdb", source)
    pg("createdb", target)
    version = sql(source, "SHOW server_version")
    # All application tables plus a marker binding the SQL/file checkpoint.
    env["DATABASE_URL"] = (
        f"postgresql+asyncpg://{args.user}:test@127.0.0.1:{args.port}/{source}"
    )
    subprocess.run(
        [sys.executable, "-m", "tools.release_restore_probe", "--create-schema"],
        env=env,
        check=True,
        capture_output=True,
    )
    sql(
        source,
        "CREATE TABLE release_probe_checkpoint(id text PRIMARY KEY); "
        "INSERT INTO release_probe_checkpoint VALUES ('checkpoint')",
    )
    graph(root / "source-kb", "write")
    snapshot_at = datetime.now(UTC)
    pg("pg_dump", "-Fc", "-d", source, "-f", str(root / "checkpoint.dump"))
    shutil.copytree(root / "source-kb", root / "checkpoint-kb")
    original_hashes = hashes(root / "checkpoint-kb")
    sql(source, "INSERT INTO release_probe_checkpoint VALUES ('after-checkpoint')")
    graph(root / "source-kb", "append")
    if sql(source, "SELECT count(*) FROM release_probe_checkpoint") != "2":
        raise RuntimeError("Post-checkpoint fixture write missing")
    restore_at = datetime.now(UTC)
    started = perf_counter()
    pg(
        "pg_restore",
        "--exit-on-error",
        "--no-owner",
        "-d",
        target,
        str(root / "checkpoint.dump"),
    )
    sql_ms = (perf_counter() - started) * 1000
    shutil.copytree(root / "checkpoint-kb", root / "restored-kb")
    copied_ms = (perf_counter() - started) * 1000
    matching = original_hashes == hashes(root / "restored-kb")
    recovered = sql(target, "SELECT string_agg(id, ',') FROM release_probe_checkpoint")
    if not matching or recovered != "checkpoint":
        raise RuntimeError("SQL/LightRAG checkpoint mismatch")
    graph(root / "restored-kb", "verify")
    rto_ms = (perf_counter() - started) * 1000
    finished_at = datetime.now(UTC).isoformat()
    report = {
        "schema_version": 1,
        "scope": "disposable local fixture, not production RTO/RPO",
        "started_at": restore_at.isoformat(),
        "finished_at": finished_at,
        "postgres_version": version,
        "postgres_tables": int(
            sql(
                target,
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public'",
            )
        ),
        "sql_restore_ms": sql_ms,
        "sql_and_copy_ms": copied_ms,
        "restore_and_verification_ms": rto_ms,
        "sql_dump_bytes": (root / "checkpoint.dump").stat().st_size,
        "kb_bytes": sum(
            p.stat().st_size for p in (root / "checkpoint-kb").rglob("*") if p.is_file()
        ),
        "kb_files": len(original_hashes),
        "file_hashes_match": matching,
        "restored_sql_marker": True,
        "restored_vector_retrieval": True,
        "checkpoint_age_at_restore_start_seconds": (
            restore_at - snapshot_at
        ).total_seconds(),
        "source_pairs": 2,
        "recovered_pairs": 1,
        "post_checkpoint_pairs_not_restored": 1,
        "provider_network_calls": 0,
        "kb_checkpoint_hashes": original_hashes,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


async def create_schema() -> None:
    from db.postgres.db import engine
    from db.postgres.models import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


def main() -> None:
    logging.disable(logging.CRITICAL)
    if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
        raise ValueError("Disable dotenv before execution")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-bin", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--user", default="abitur_390")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--mode", choices=("write", "append", "verify"))
    parser.add_argument("--create-schema", action="store_true")
    args = parser.parse_args()
    if args.create_schema:
        asyncio.run(create_schema())
    elif args.graph:
        asyncio.run(graph_fixture(args.graph, args.mode))
    else:
        run_fixture(args)


if __name__ == "__main__":
    main()
