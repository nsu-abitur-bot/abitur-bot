"""Pure calculations for explicit numeric measurements, with unknowns preserved."""

import math
import statistics
from collections import defaultdict
from decimal import Decimal
from typing import Any


def request_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Failed and timed-out requests stay in the denominator."""
    if not rows:
        return {"total": 0, "error_rate": None, "p50_ms": None, "p95_ms": None}
    durations = []
    for row in rows:
        value = row["duration_ms"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or row["status"] not in {"ok", "error", "timeout"}
        ):
            raise ValueError("Invalid request observation")
        durations.append(value)
    durations.sort()
    failed = sum(row["status"] != "ok" for row in rows)
    return {
        "total": len(rows),
        "failed": failed,
        "error_rate": failed / len(rows),
        "p50_ms": statistics.median(durations),
        "p95_ms": durations[math.ceil(0.95 * len(rows)) - 1],
        "over_30_seconds": sum(value > 30_000 for value in durations),
        "percentile_method": "p50 median; p95 nearest rank; all terminal outcomes",
    }


def usage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per billable attempt; a missing retry or token count is unknown."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row["attempt_id"] in seen:
            raise ValueError("Duplicate attempt")
        seen.add(row["attempt_id"])
        if row["purpose"] not in {"question", "indexing", "judge"}:
            raise ValueError("Invalid purpose")
        if row["kind"] not in {"generation", "embedding"}:
            raise ValueError("Invalid operation kind")
        for field in ("input_tokens", "output_tokens"):
            value = row.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Invalid token count")
        groups[(row["purpose"], row["provider"], row["model"], row["kind"])].append(row)
    result = []
    for key, values in sorted(groups.items()):
        cost = Decimal(0)
        complete = True
        currencies = set()
        for row in values:
            tokens = (row.get("input_tokens"), row.get("output_tokens"))
            rates = (
                row.get("input_rate_per_million"),
                row.get("output_rate_per_million"),
            )
            if None in tokens or None in rates or not row.get("price_source"):
                complete = False
                continue
            parsed = [Decimal(str(rate)) for rate in rates]
            if any(not rate.is_finite() or rate < 0 for rate in parsed):
                raise ValueError("Invalid price")
            if not row.get("currency") or not row.get("price_checked_at"):
                raise ValueError("Price requires currency and date")
            currencies.add(row["currency"])
            cost += sum(
                (Decimal(str(t)) * r for t, r in zip(tokens, parsed)), Decimal(0)
            ) / Decimal(1_000_000)
        if len(currencies) > 1:
            raise ValueError("Mixed currencies in one model group")
        sums = {}
        for field in ("input_tokens", "output_tokens"):
            sums[field] = (
                sum(row[field] for row in values)
                if all(row.get(field) is not None for row in values)
                else None
            )
        result.append(
            dict(
                zip(("purpose", "provider", "model", "kind"), key),
                observed_attempts=len(values),
                **sums,
                cost=str(cost) if complete else None,
                currency=next(iter(currencies)) if currencies else None,
                coverage="observed attempts only; caller must prove capture completeness",
            )
        )
    return result


def availability_summary(
    *, period_seconds: float, available_seconds: float, unavailable_seconds: float
) -> dict[str, float | None]:
    values = (period_seconds, available_seconds, unavailable_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values) or period_seconds == 0:
        raise ValueError("Invalid observation interval")
    covered = available_seconds + unavailable_seconds
    if covered > period_seconds:
        raise ValueError("Overlapping availability intervals")
    missing = period_seconds - covered
    return {
        "period_seconds": period_seconds,
        "unobserved_seconds": missing,
        "availability_percent": 100 * available_seconds / period_seconds
        if missing == 0
        else None,
    }
