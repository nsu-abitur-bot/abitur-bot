"""One Cloud.ru retry through the operator meter, under the original owner caps."""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from tools.release470_cloudru import HOST, MODEL, CloudCounter, cost_micro_rub
from tools.release470_store import BudgetStore

BUDGET = Path("/reports/release470-budget.sqlite3")
OWNER_CAPS = {"micro_rub": 1000000, "micro_cny": 3000000, "embedding": 10000}


def check_owner_caps(state: dict, tokens: int, rate: int) -> int:
    if state["halted"] or state["pending_or_uncertain"] or rate != 3222:
        raise RuntimeError("Budget or tariff is not ready for this retry")
    cost = cost_micro_rub(tokens, rate)
    additions = {"micro_rub": cost, "micro_cny": 0, "embedding": tokens}
    if any(
        state["totals"][key] + additions[key] > cap for key, cap in OWNER_CAPS.items()
    ):
        raise RuntimeError("The original owner limit would be exceeded")
    return cost


@contextlib.contextmanager
def one_cloud_request(
    payload: dict, state: Callable[[], dict], tokens: int, rate: int, result: dict
) -> Iterator[None]:
    """Add a stricter outer guard; retain the existing root meter unchanged."""
    original = httpx.Client.send
    used = False

    def send(client, request: httpx.Request, **kwargs: Any):
        nonlocal used
        if (
            used
            or request.method != "POST"
            or request.url.scheme != "https"
            or request.url.host != HOST
            or request.url.path != "/v1/embeddings"
            or request.url.port not in (None, 443)
            or request.url.query
            or request.url.userinfo
            or json.loads(request.content) != payload
        ):
            raise RuntimeError("Only one exact approved Cloud.ru attempt is allowed")
        check_owner_caps(state(), tokens, rate)
        used = True
        result["physical_attempts"] = 1
        response = original(client, request, **kwargs)
        body = response.json()
        result["http_status"] = response.status_code
        result["response_model"] = body.get("model")
        usage = body.get("usage", {})
        result["usage"] = {
            key: value
            for key, value in usage.items()
            if key in {"prompt_tokens", "total_tokens"} and type(value) is int
        }
        return response

    async def deny_async(*args: Any, **kwargs: Any):
        raise RuntimeError("This check has no asynchronous provider calls")

    with (
        patch.object(httpx.Client, "send", send),
        patch.object(httpx.AsyncClient, "send", deny_async),
    ):
        yield


async def check(result: dict) -> None:
    from bootstrap.runtime_settings import runtime_settings
    from config.instance import get_setting
    from db.postgres.db import engine
    from llm.factory import get_faq_provider

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected isolated target")
    plan = json.loads(Path("/reports/pilot-plan-approved-2026-11-21.json").read_text())
    texts = [plan["questions"][key] for key in ("faq-1", "faq-2")]
    store = BudgetStore(BUDGET)
    for key, text in zip(("faq-1", "faq-2"), texts):
        if hashlib.sha256(text.encode()).hexdigest() != store.policy["case_plan"][key]:
            raise RuntimeError("FAQ text differs from the agreed plan")
    payload = {"model": MODEL, "input": texts, "encoding_format": "float"}
    counter = CloudCounter(
        Path("/evaluation/cloudru-tokenizer.json"), store.policy["cloudru"]
    )
    tokens = counter(payload)
    rate = store.policy["cloudru"]["rate_hundredths"]
    before = store.snapshot()
    result["budget_before"] = before
    result["reserved_micro_rub_bound"] = check_owner_caps(before, tokens, rate)
    result["token_bound"] = tokens
    async with runtime_settings():
        if get_setting("FAQ_EMBEDDING_PROVIDER") != "cloudru":
            raise RuntimeError("Unexpected FAQ embedding provider")
        with one_cloud_request(payload, store.snapshot, tokens, rate, result):
            embedder = get_faq_provider().get_embeddings_model()
            if embedder is None:
                raise RuntimeError("FAQ embedding model is unavailable")
            vectors = await asyncio.to_thread(embedder.embed_documents, texts)
        after = store.snapshot()
        usage = result.get("usage", {})
        if (
            len(vectors) != 2
            or any(len(vector) != 2048 for vector in vectors)
            or result.get("http_status") != 200
            or result.get("response_model") != MODEL
            or usage.get("prompt_tokens") != tokens
            or usage.get("total_tokens") != tokens
            or after["attempts_started"] != before["attempts_started"] + 1
            or after["totals"]["micro_cny"] != before["totals"]["micro_cny"]
        ):
            raise RuntimeError(
                "The single-call result does not match the reviewed contract"
            )
        check_owner_caps(after, 0, rate)
        result["budget_after"] = after
        result["vector_count"] = 2
        result["dimensions"] = 2048
        result["completed"] = True


def main() -> None:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    logging.disable(logging.CRITICAL)
    output = Path("/reports/cloudru-recheck-20261123.json")
    result: dict = {
        "completed": False,
        "physical_attempts": 0,
        "messages_sent": 0,
        "index_changed": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    with output.open("x") as stream:
        json.dump(result, stream)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            asyncio.run(check(result))
    except BaseException as error:
        result["error_type"] = type(error).__name__
        chain = []
        cause = error.__cause__ or error.__context__
        while cause is not None and len(chain) < 5:
            chain.append(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        result["cause_types"] = chain
        with contextlib.suppress(Exception):
            result["budget_after"] = BudgetStore(BUDGET).snapshot()
    result["finished_at_utc"] = datetime.now(UTC).isoformat()
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
