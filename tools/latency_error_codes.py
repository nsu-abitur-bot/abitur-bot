"""Extract numeric Google error codes from known static application error prefixes.

Run on the server against a bounded cohort window, only after independent review.
Never export exception bodies, session identifiers or arbitrary status strings.
"""

import json
import re
import sys
from datetime import datetime

_PREFIX = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
    r"(pipeline\.llm_client|rag\.crag|llm\.providers\.gemini_graph_adapters) - "
    r"(?:WARNING|ERROR) - (.*)$"
)
_PATTERNS = (
    (
        "pipeline.llm_client",
        "generation",
        r"\[[^\]\r\n]*\] Provider generation error: ([45]\d{2}) (?:[A-Z_]+|None)\.",
    ),
    (
        "rag.crag",
        "crag_grade",
        r"CRAG: ошибка грейдинга чанка #\d+: ([45]\d{2}) (?:[A-Z_]+|None)\.",
    ),
    (
        "llm.providers.gemini_graph_adapters",
        "graph_llm",
        r"Error in GeminiLLM: ([45]\d{2}) (?:[A-Z_]+|None)\.",
    ),
    (
        "llm.providers.gemini_graph_adapters",
        "graph_embedding",
        r"Error in GeminiEmbedding: ([45]\d{2}) (?:[A-Z_]+|None)\.",
    ),
)


def extract(line: str) -> dict | None:
    """Return only validated timestamp, static component and a numeric error code."""
    match = _PREFIX.fullmatch(line.rstrip("\n"))
    if match is None:
        return None
    try:
        datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    for logger, component, pattern in _PATTERNS:
        if logger == match[2]:
            error = re.match(pattern, match[3])
            if error is not None:
                return {
                    "logged_at_utc": match[1],
                    "component": component,
                    "reported_http_code": int(error[1]),
                }
    return None


def main() -> None:
    """Discard all logs except the numeric schema above."""
    for line in sys.stdin:
        row = extract(line)
        if row is not None:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
