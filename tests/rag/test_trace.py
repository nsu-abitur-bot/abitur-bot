"""Трасса внутренних логов LightRAG без гонок между запросами (#303).

Прежний механизм — временный хендлер на процессном логгере lightrag с
восстановлением level/propagate в finally — при двух одновременных запросах
смешивал строки чужих трасс и оставлял «залипший» уровень. Тесты ниже
закрепляют свойства замены: контекстную изоляцию и отсутствие мутаций
конфигурации логгера.
"""

import asyncio
import logging

from rag.trace import (
    LIGHTRAG_LOGGER_NAME,
    capture_lightrag_trace,
    install_lightrag_trace_handler,
)


def _emit(line: str) -> None:
    logging.getLogger(LIGHTRAG_LOGGER_NAME).info(line)


async def test_concurrent_traces_stay_separate():
    """Строки соседнего запроса не попадают в чужую трассу."""

    async def request(marker: str) -> list[str]:
        with capture_lightrag_trace() as trace:
            for i in range(20):
                _emit(f"{marker} line {i}")
                await asyncio.sleep(0)
        return trace.get_lines()

    first, second = await asyncio.gather(request("FIRST"), request("SECOND"))

    assert first and all("FIRST" in line for line in first), f"чужие строки: {first}"
    assert second and all("SECOND" in line for line in second), f"чужие строки: {second}"


async def test_records_after_capture_are_not_routed():
    with capture_lightrag_trace() as trace:
        _emit("внутри захвата")
    _emit("после выхода")

    lines = trace.get_lines()
    assert len(lines) == 1
    assert "внутри захвата" in lines[0]


async def test_stale_task_does_not_leak_into_live_trace():
    """Задача, переживший свою трассу, не загрязняет трассу нового запроса.

    Воркер создан внутри первой трассы и стреляет позже, когда открыта вторая:
    осиротевшая строка может достаться только закрытому буферу первой трассы,
    но не живой второй. LightRAG держит пул воркеров между запросами, так что
    этот случай реален, а не теоретичен.
    """
    release = asyncio.Event()

    async def stale_worker() -> None:
        await release.wait()
        _emit("stale worker line")

    with capture_lightrag_trace() as first:
        stale_task = asyncio.create_task(stale_worker())
        _emit("first request line")

    with capture_lightrag_trace() as second:
        release.set()
        await stale_task
        _emit("second request line")

    assert second.get_lines() == ["INFO: second request line"]
    # Буфер первой трассы уже никем не читается: осиротевшие строки в нём
    # безвредны, состояние второй трассы от них не зависит.
    assert "second request line" not in "\n".join(first.get_lines())


def test_capture_does_not_touch_logger_configuration():
    logger = logging.getLogger(LIGHTRAG_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    handlers_before = list(logger.handlers)
    propagate_before = logger.propagate

    with capture_lightrag_trace() as trace:
        _emit("captured line")
        # Внутри захвата хендлеры не перекладываются и propagate не трогается.
        assert list(logger.handlers) == handlers_before
        assert logger.propagate is propagate_before

    assert list(logger.handlers) == handlers_before
    assert logger.propagate is propagate_before
    assert "captured line" in trace.get_lines()[0]


def test_install_is_idempotent_and_only_lowers_level():
    logger = logging.getLogger(LIGHTRAG_LOGGER_NAME)
    logger.setLevel(logging.CRITICAL)

    install_lightrag_trace_handler()
    handlers_after_first = list(logger.handlers)

    install_lightrag_trace_handler()
    # Повторная установка не плодит хендлеры и не меняет уровень.
    assert list(logger.handlers) == handlers_after_first
    # INFO — рабочий уровень lightrag из logging_config; строже ему быть нельзя,
    # иначе записи не дойдут до хендлера и трасса опустеет.
    assert logger.level == logging.INFO
