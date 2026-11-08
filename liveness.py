"""HTTP-проба живости для процессов без своего HTTP-сервера (#241).

Бот (long polling) и планировщик (цикл синхронизации) не открывают портов:
снаружи не отличить работающий процесс от зависшего. Этот сервер отвечает,
пока жив event loop процесса, и используется docker-compose healthcheck
внутри того же контейнера. Наружу порт не публикуется — слушается только
127.0.0.1, поэтому проба не расширяет поверхность атаки.
"""

import logging

from aiohttp import web

from config.instance import get_setting

logger = logging.getLogger(__name__)

LIVENESS_HOST = "127.0.0.1"
DEFAULT_LIVENESS_PORT = 8080


async def _ok(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


class LivenessServer:
    """Минимальный сервер с единственным маршрутом GET /healthz.

    port=0 выделяет свободный порт ОС — так тесты не конфликтуют между собой.
    """

    def __init__(self, port: int | None = None) -> None:
        self._requested_port = (
            port
            if port is not None
            else int(get_setting("LIVENESS_PORT", str(DEFAULT_LIVENESS_PORT)))
        )
        self._runner: web.AppRunner | None = None

    @property
    def port(self) -> int:
        """Фактический порт: после старта с port=0 — выделенный ОС."""
        addresses = self._runner.addresses if self._runner else []
        return addresses[0][1] if addresses else self._requested_port

    async def start(self) -> None:
        """Запускает пробу; занятый порт не роняет процесс — это диагностика,
        а не критичная функция."""
        app = web.Application()
        app.router.add_get("/healthz", _ok)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, LIVENESS_HOST, self._requested_port)
            await site.start()
        except OSError as exc:
            # Например, два процесса на одной машине вне compose с общим
            # LIVENESS_PORT. Процесс продолжит работу без пробы.
            await runner.cleanup()
            logger.warning(
                "Проба живости не поднялась на порту %d: %s",
                self._requested_port,
                exc,
            )
            return
        self._runner = runner
        logger.info("Проба живости: http://%s:%d/healthz", LIVENESS_HOST, self.port)

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
