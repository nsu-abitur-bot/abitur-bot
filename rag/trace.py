"""Сбор внутренних логов LightRAG в трассу конкретного запроса.

LightRAG сообщает о своих действиях (извлечение ключевых слов, поиск чанков,
вызовы модели) только через процессный логгер ``lightrag``. Раньше оркестратор
собирал из этого трассу временным хендлером: поставить на время запроса,
изменить level/propagate, восстановить в finally. При конкурентных запросах
хендлеры смешивали строки чужих запросов, а восстановление состояния
«залипало» (#303).

Теперь один постоянный хендлер роутит записи по ContextVar: активная трасса
видна только контексту выполнения, который её открыл, — соседние запросы
работают параллельно, не задевая друг друга. Хендлер ставится один раз и
никогда не снимается; propagate логгера не трогаем вовсе, а уровень только
понижаем до INFO (столько же, сколько задаёт logging_config), и один раз.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Generator, Optional, TypeVar

from config.instance import SettingsBinding, capture_settings, settings_scope

LIGHTRAG_LOGGER_NAME = "lightrag"
RAG_INTERNAL_LOG_LIMIT = 120
T = TypeVar("T")


class LightragTrace:
    """Буфер строк трассы с ограничением по количеству."""

    def __init__(self, limit: int = RAG_INTERNAL_LOG_LIMIT) -> None:
        self.limit = limit
        self.lines: list[str] = []
        self.dropped = 0
        self.closed = False

    def add(self, record: logging.LogRecord) -> None:
        try:
            line = f"{record.levelname}: {record.getMessage()}"
        except Exception:
            return
        self.note(line)

    def note(self, line: str) -> None:
        """Добавляет диагностическую строку самого ретривера."""
        if self.closed:
            return
        if len(self.lines) < self.limit:
            self.lines.append(line)
        else:
            self.dropped += 1

    def get_lines(self) -> list[str]:
        if self.dropped <= 0:
            return list(self.lines)
        return [*self.lines, f"... [truncated {self.dropped} internal RAG log lines]"]


@dataclass
class RetrievalResult:
    """Результат ретрива: контекст, источники и трасса ретривера.

    Трасса приходит в результате, а не через перехват процессного логгера
    в вызывающем коде: там она смешивалась между конкурентными запросами (#303).
    """

    context: str
    sources: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    answer: str | None = None
    generation_question: str | None = None
    evidence: list[str] = field(default_factory=list)


_active_trace: ContextVar[Optional[LightragTrace]] = ContextVar(
    "lightrag_trace", default=None
)


class _RoutingHandler(logging.Handler):
    """Пишет записи lightrag в трассу текущего контекста, если она открыта."""

    def emit(self, record: logging.LogRecord) -> None:
        trace = _active_trace.get()
        if trace is not None:
            trace.add(record)


_handler: Optional[_RoutingHandler] = None


def install_lightrag_trace_handler() -> None:
    """Идемпотентно ставит роутящий хендлер на логгер lightrag.

    Отдельные запросы этого не делают: хендлер один на процесс, снятия и
    восстановления состояния нет — восстанавливать нечего, и гонок нет.
    """
    global _handler
    lightrag_logger = logging.getLogger(LIGHTRAG_LOGGER_NAME)
    # INFO — рабочий уровень lightrag из logging_config; строже нельзя, иначе
    # записи не дойдут до хендлера и трасса опустеет. Проверяем при каждом
    # вызове: уровень могли ужесточить и после установки хендлера.
    if lightrag_logger.level == logging.NOTSET or lightrag_logger.level > logging.INFO:
        lightrag_logger.setLevel(logging.INFO)
    if _handler is None:
        _handler = _RoutingHandler(level=logging.INFO)
        lightrag_logger.addHandler(_handler)


@contextmanager
def capture_lightrag_trace() -> Generator[LightragTrace, None, None]:
    """Открывает трассу для текущего контекста выполнения.

    Записи lightrag, сделанные в этом контексте (задача asyncio и её потомки),
    попадают в возвращённую трассу; записи других контекстов — в их трассы.
    После выхода контекст видит прежнее значение, «хвостов» не остаётся.
    """
    install_lightrag_trace_handler()
    trace = LightragTrace()
    token = _active_trace.set(trace)
    try:
        yield trace
    finally:
        trace.closed = True
        _active_trace.reset(token)


@contextmanager
def _bind_trace(trace: LightragTrace | None) -> Generator[None, None, None]:
    token = _active_trace.set(trace)
    try:
        yield
    finally:
        _active_trace.reset(token)


def trace_worker_callback(
    function: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Bind the trace carried by a queue job, including wait_for child tasks."""

    @wraps(function)
    async def wrapped(
        *args,
        _lightrag_trace: LightragTrace | None = None,
        _lightrag_settings: SettingsBinding | None = None,
        **kwargs,
    ) -> T:
        binding = _lightrag_settings or capture_settings()
        with (
            _bind_trace(_lightrag_trace),
            settings_scope(binding.values, binding.revision),
        ):
            return await function(*args, **kwargs)

    return wrapped


def trace_queued_callback(
    function: Callable[..., Awaitable[T]], name: str
) -> Callable[..., Awaitable[T]]:
    """Keep persistent workers neutral; deliver their failure at the caller.

    LightRAG's queue does not transfer context per job. Its workers and health
    checks must not inherit the first request's trace. The explicit job argument
    is consumed by trace_worker_callback before reaching a provider. Queue errors
    (including execution/queue timeouts) are recorded here, where the owner is
    known; the queue's own housekeeping logs remain in the process log.
    """

    @wraps(function)
    async def wrapped(*args, **kwargs) -> T:
        trace = _active_trace.get()
        binding = capture_settings()
        try:
            with _bind_trace(None), settings_scope(None, 0):
                return await function(
                    *args, _lightrag_trace=trace, _lightrag_settings=binding, **kwargs
                )
        except Exception as exc:
            if trace is not None:
                trace.add(
                    logging.getLogger(LIGHTRAG_LOGGER_NAME).makeRecord(
                        LIGHTRAG_LOGGER_NAME,
                        logging.ERROR,
                        __file__,
                        0,
                        "%s queued call failed: %s: %s",
                        (name, type(exc).__name__, str(exc)),
                        None,
                    )
                )
            raise

    return wrapped
