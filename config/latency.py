"""Content-free monotonic timings, correlated across tasks and worker threads."""

import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any, Awaitable, Callable, Coroutine, Generator, ParamSpec, TypeVar
from uuid import uuid4

logger = logging.getLogger("abitur.latency")
_request_id: ContextVar[str | None] = ContextVar("latency_request_id", default=None)
_first_request = True
P = ParamSpec("P")
T = TypeVar("T")


def observation(name: str, duration_ms: float) -> None:
    """Record a precomputed duration (e.g. Telegram timestamp to handler entry)."""
    logger.info(
        "latency %s",
        json.dumps(
            {
                "request_id": _request_id.get(),
                "stage": name,
                "duration_ms": round(duration_ms, 3),
                "status": "observed",
                "error_type": None,
            }
        ),
    )


@contextmanager
def stage(name: str) -> Generator[None, None, None]:
    """Record wall time, including failed/cancelled operations, without payloads."""
    started = perf_counter()
    status = "ok"
    error_type = None
    try:
        yield
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        error_type = type(exc).__name__
        raise
    finally:
        logger.info(
            "latency %s",
            json.dumps(
                {
                    "request_id": _request_id.get(),
                    "stage": name,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "status": status,
                    "error_type": error_type,
                }
            ),
        )


def timed(
    name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Time an async boundary; names must be static, never derived from input."""

    def decorate(
        function: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            with stage(name):
                return await function(*args, **kwargs)

        return wrapped

    return decorate


def request_timed(
    name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Give an outer request a random ID; nested pipeline calls share it."""

    def decorate(
        function: Callable[P, Awaitable[T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            global _first_request
            if _request_id.get() is not None:
                with stage(name):
                    return await function(*args, **kwargs)
            token = _request_id.set(uuid4().hex)
            first = _first_request
            _first_request = False
            try:
                with stage(name):
                    logger.info(
                        "latency_request %s",
                        json.dumps(
                            {"request_id": _request_id.get(), "process_first": first}
                        ),
                    )
                    return await function(*args, **kwargs)
            finally:
                _request_id.reset(token)

        return wrapped

    return decorate
