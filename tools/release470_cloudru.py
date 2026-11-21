"""Cloud.ru embedding accounting, without network, credentials or remote code."""

import hashlib
import math
from importlib import import_module
from pathlib import Path

from tools.admission_usage import IndexingLimit, complete_usage, usage_counts

HOST = "foundation-models.api.cloud.ru"
MODEL = "ai-sage/Giga-Embeddings-instruct-3B"
DIMENSIONS = 2048


def cost_micro_rub(tokens: int, rate_hundredths: int) -> int:
    """RUB per million tokens -> micro-RUB, rounded up per physical attempt."""
    if (
        type(tokens) is not int
        or tokens < 0
        or type(rate_hundredths) is not int
        or rate_hundredths <= 0
    ):
        raise IndexingLimit("Invalid RUB rate or token usage")
    return (tokens * rate_hundredths + 99) // 100


class CloudCounter:
    """Operator-reviewed mapping of the served model to a local public tokenizer.

    This never guesses that a similarly named Hugging Face model is identical.
    The operator must supply the binding after reviewing the actual Cloud.ru model.
    """

    def __init__(self, path: Path, settings: dict) -> None:
        if path.name != "cloudru-tokenizer.json" or path.is_symlink():
            raise IndexingLimit("A fixed public Cloud.ru tokenizer artifact is required")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != settings["tokenizer_sha256"]:
            raise IndexingLimit("Cloud.ru tokenizer integrity check failed")
        self.tokenizer = import_module("tokenizers").Tokenizer.from_str(raw.decode())
        # Count the entire input even if the artifact has saved truncation settings.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.settings = settings

    @property
    def evidence(self) -> dict:
        return {
            "kind": "reviewed-cloudru-tokenizer",
            "model": MODEL,
            "sha256": self.settings["tokenizer_sha256"],
            "revision": self.settings["tokenizer_revision"],
        }

    def __call__(self, payload: dict) -> int:
        total = 0
        for value in payload["input"]:
            # Special-token postprocessing is part of the reviewed JSON artifact.
            count = len(self.tokenizer.encode(value, add_special_tokens=True).ids)
            if not 1 <= count <= 8192:
                raise IndexingLimit("Cloud.ru per-input context bound exceeded")
            total += count
        return total


def validate_request(payload: dict) -> None:
    if (
        not isinstance(payload, dict)
        or set(payload) - {"model", "input", "encoding_format"}
        or payload.get("model") != MODEL
        or payload.get("encoding_format") != "float"
    ):
        raise IndexingLimit("Only the reviewed Cloud.ru embedding request is allowed")
    texts = payload.get("input")
    if (
        not isinstance(texts, list)
        or not 1 <= len(texts) <= 16
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise IndexingLimit("A bounded nonempty Cloud.ru text batch is required")


def final_usage(body: dict, expected_count: int) -> tuple[dict, bool, bool]:
    if not isinstance(body, dict):
        return {}, False, False
    # Embedding usage has the same prompt/total semantics as the existing helper.
    usage = usage_counts(body, "gigachat")
    usage_valid = complete_usage(usage, "gigachat")
    raw = body.get("usage")
    if (
        not isinstance(raw, dict)
        or type(raw.get("prompt_tokens")) is not int
        or type(raw.get("total_tokens")) is not int
        or raw["prompt_tokens"] <= 0
        or any(
            type(v) is not int or v < 0
            for k, v in raw.items()
            if k in {"prompt_tokens", "total_tokens", "completion_tokens"}
        )
    ):
        usage_valid = False
    rows = body.get("data")
    valid = (
        usage_valid
        and body.get("model") == MODEL
        and isinstance(rows, list)
        and len(rows) == expected_count
    )
    seen = set()
    if not valid:
        return usage, usage_valid, False
    for row in rows:
        if not isinstance(row, dict):
            return usage, usage_valid, False
        index, vector = row.get("index"), row.get("embedding")
        if (
            type(index) is not int
            or not 0 <= index < expected_count
            or index in seen
            or not isinstance(vector, list)
            or len(vector) != DIMENSIONS
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)
            or not any(x != 0 for x in vector)
        ):
            return usage, usage_valid, False
        seen.add(index)
    return usage, usage_valid, bool(valid)
