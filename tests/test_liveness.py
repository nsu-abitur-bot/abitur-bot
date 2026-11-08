"""Проба живости процессов без HTTP-порта (#241).

Бот и планировщик не открывают портов: liveness.py даёт compose-healthcheck
единственный маршрут GET /healthz на 127.0.0.1. Проверяем, что проба
отвечает, закрывается и не роняет процесс при занятом порту.
"""

import socket

from aiohttp import ClientSession

from liveness import LivenessServer


async def _get(port: int, path: str = "/healthz"):
    async with (
        ClientSession() as session,
        session.get(f"http://127.0.0.1:{port}{path}") as response,
    ):
        return response.status, await response.json()


async def test_healthz_answers_ok():
    server = LivenessServer(port=0)
    await server.start()

    try:
        status, body = await _get(server.port)
    finally:
        await server.aclose()

    assert status == 200
    assert body == {"status": "ok"}


async def test_aclose_stops_answering():
    server = LivenessServer(port=0)
    await server.start()
    port = server.port
    await server.aclose()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0


async def test_occupied_port_is_not_fatal():
    busy = LivenessServer(port=0)
    await busy.start()
    occupied_port = busy.port

    duplicate = LivenessServer(port=occupied_port)
    # Не поднимает исключение: проба — диагностика, а не критичная функция.
    await duplicate.start()

    try:
        status, body = await _get(occupied_port)
        assert status == 200
        assert body == {"status": "ok"}
    finally:
        await busy.aclose()
        await duplicate.aclose()


async def test_unknown_path_returns_404():
    server = LivenessServer(port=0)
    await server.start()

    try:
        async with (
            ClientSession() as session,
            session.get(f"http://127.0.0.1:{server.port}/") as response,
        ):
            assert response.status == 404
    finally:
        await server.aclose()
