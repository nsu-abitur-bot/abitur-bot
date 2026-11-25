"""Read-only checks of the agreed public #470 seed; no app imports or network."""

import hashlib
import json
from pathlib import Path

CHUNKS_SHA256 = "90fab4be507b675a92f5220a22e756f3c4db8a825bc5b571a143080108db008e"
TOKENIZER_SHA256 = "6fb1280bd7fd529f425929b5df823a5a44485cd7fa9679d1ec1acaac4962e8ca"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe() -> dict:
    candidates = [
        Path("/app/data/lightrag/abitur_kb/kv_store_text_chunks.json"),
        Path("/data/lightrag/abitur_kb/kv_store_text_chunks.json"),
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError("Agreed seed is not at a reviewed location")
    path = existing[0]
    if path.is_symlink() or path.stat().st_size > 5_000_000:
        raise ValueError("Unexpected seed file")
    actual = digest(path)
    if actual != CHUNKS_SHA256:
        raise ValueError("Seed differs from the operator-verified snapshot")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise ValueError("Unexpected chunks format")
    contents = [str(row.get("content", "")).casefold() for row in rows.values()]
    tokenizer = Path("/evaluation/cloudru-tokenizer.json")
    tokenizer_ok = tokenizer.is_file() and digest(tokenizer) == TOKENIZER_SHA256
    result = {
        "scope": "read-only public seed file hashes and topic presence",
        "chunks_sha256": actual,
        "chunk_count": len(rows),
        "tokenizer_sha256_matches": tokenizer_ok,
        "topic_presence_only": {
            "profiles": sum("профил" in text for text in contents),
            "programming_languages": sum(
                any(word in text for word in ("python", "java", "c++", "язык"))
                for text in contents
            ),
        },
        "answer_facts_verified": False,
        "pilot_budget_exists": Path("/reports/release470-budget.sqlite3").exists(),
        "provider_calls": 0,
        "messages_sent": 0,
        "storage_writes": 0,
    }
    if not tokenizer_ok or len(rows) != 82:
        raise ValueError("Prepared seed/tokenizer checks failed")
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(probe(), ensure_ascii=True, indent=2))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(json.dumps({"preflight_failed": type(error).__name__}))
        raise SystemExit(1) from None
