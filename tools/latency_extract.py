"""Extract validated content-free latency records from application logs on stdin.

Run filtering on the server before transferring output. Never save unfiltered logs.
Only the configured abitur.latency logger prefix and registered schemas are read.
"""

import json
import math
import re
import sys
from typing import Any

STAGES = frozenset(
    {
        "telegram_update_age",
        "telegram_total",
        "telegram_typing",
        "telegram_draft",
        "telegram_send",
        "pipeline_total",
        "history_write",
        "history",
        "faq",
        "faq_embedding",
        "retrieval_total",
        "graph_acquire",
        "search_embeddings",
        "rag_embedding",
        "rag_model_call",
        "source_titles",
        "source_ranking",
        "source_ranking_model_call",
        "crag_total",
        "crag_filter",
        "crag_grade",
        "crag_model_call",
        "crag_refine",
        "crag_refine_model_call",
        "generation_tools_total",
        "generation_first_delta",
        "tool_execution",
        "answer_total",
        "intent_background",
        "intent_model_call",
    }
)
ERROR_TYPES = frozenset(
    {
        "CancelledError",
        "ConnectionError",
        "TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ClientError",
        "ServerError",
        "APIError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "HTTPStatusError",
        "RetrievalError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "ValidationError",
        "OperationalError",
        "ProgrammingError",
        "IntegrityError",
        "OSError",
        "TelegramBadRequest",
        "TelegramNetworkError",
        "TelegramRetryAfter",
    }
)
PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - abitur\.latency - INFO - "
    r"(latency(?:_request)?) (\{.*\})\s*$"
)


def extract(line: str) -> dict[str, Any] | None:
    """Drop malformed/unregistered fields, rather than treating text as telemetry."""
    match = PREFIX.fullmatch(line.rstrip("\n"))
    if match is None:
        return None
    try:
        row = json.loads(match[2])
    except (ValueError, TypeError):
        return None
    if not isinstance(row, dict):
        return None
    request_id = row.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id)
    ):
        return None
    if match[1] == "latency_request":
        if not isinstance(row.get("process_first"), bool):
            return None
        return {"request_id": request_id, "process_first": row["process_first"]}
    stage = row.get("stage")
    status = row.get("status")
    duration = row.get("duration_ms")
    if not isinstance(stage, str) or stage not in STAGES:
        return None
    if not isinstance(status, str) or status not in {
        "ok",
        "error",
        "cancelled",
        "observed",
    }:
        return None
    if type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
        return None
    if "process_first" in row:
        return None
    error_type = row.get("error_type")
    if error_type is not None:
        if not isinstance(error_type, str):
            return None
        error_type = error_type if error_type in ERROR_TYPES else "OtherError"
    return {
        "request_id": request_id,
        "stage": stage,
        "duration_ms": duration,
        "status": status,
        "error_type": error_type,
    }


def main() -> None:
    """Filter stdin; malformed and unrelated records are silently discarded."""
    for line in sys.stdin:
        row = extract(line)
        if row is not None:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
