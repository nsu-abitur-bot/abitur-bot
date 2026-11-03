"""GigaChat embeddings with scoped TLS trust and renewable OAuth credentials."""

import asyncio
import hashlib
import math
import os
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4
from weakref import WeakValueDictionary

import httpx

from config.instance import get_setting
from llm.base import BaseLLMProvider, LLMResult

MODEL_DIMENSIONS = {"Embeddings": 1024, "Embeddings-2": 1024, "EmbeddingsGigaR": 2560}
OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
EMBEDDINGS_URL = "https://api.giga.chat/v1/embeddings"


@dataclass
class _RequestGate:
    lock: threading.Lock = field(default_factory=threading.Lock)


_gates: WeakValueDictionary[str, _RequestGate] = WeakValueDictionary()
_gates_guard = threading.Lock()


def _request_gate(identity: str) -> _RequestGate:
    with _gates_guard:
        gate = _gates.get(identity)
        if gate is None:
            gate = _RequestGate()
            _gates[identity] = gate
        return gate


@contextmanager
def _request_slot(identity: str) -> Generator[None]:
    # One personal account has one stream, shared by API/bot and all demo instances.
    deadline = time.monotonic() + 120
    gate = _request_gate(identity)
    if not gate.lock.acquire(timeout=120):
        raise GigaChatError("Истекло ожидание очереди эмбеддингов GigaChat")
    try:
        directory = os.getenv("GIGACHAT_LOCK_DIRECTORY")
        if not directory:
            yield
            return
        if os.name != "posix":
            raise GigaChatError("Общий каталог очереди GigaChat требует POSIX")
        import fcntl

        root = Path(directory)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            root / (identity + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as lock:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise GigaChatError(
                            "Истекло ожидание общей очереди эмбеддингов GigaChat"
                        ) from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        gate.lock.release()


class GigaChatError(RuntimeError):
    """Contains no upstream body, credential, URL or token."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(repr=False)
class _Token:
    value: str = ""
    expires_at: float = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@lru_cache(maxsize=16)
def _token_state(auth_key: str, scope: str) -> _Token:
    return _Token()


@lru_cache(maxsize=1)
def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(
        str(Path(__file__).with_name("certs") / "russian_trusted_root_ca.pem")
    )
    context.load_verify_locations(
        str(Path(__file__).with_name("certs") / "russian_trusted_sub_ca.pem")
    )
    return context


class GigaChatEmbedding:
    def __init__(
        self,
        auth_key: str | None = None,
        model: str | None = None,
        scope: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._key = auth_key or get_setting("GIGACHAT_API_KEY", "")
        self.model = model or get_setting("GIGACHAT_EMBEDDING_MODEL", "EmbeddingsGigaR")
        self._scope = scope or get_setting("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        if not self._key:
            raise GigaChatError("Не задан ключ авторизации GigaChat")
        if self.model not in MODEL_DIMENSIONS:
            raise GigaChatError("Неизвестная модель эмбеддингов GigaChat")
        if self._scope not in {
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_B2B",
            "GIGACHAT_API_CORP",
        }:
            raise GigaChatError("Неизвестная область доступа GigaChat")
        self.embedding_dim = MODEL_DIMENSIONS[self.model]
        self._transport = transport
        self._token = _Token() if transport else _token_state(self._key, self._scope)
        self._identity = hashlib.sha256(
            (self._key + "\0" + self._scope).encode()
        ).hexdigest()

    def _access_token(self, client: httpx.Client) -> str:
        with self._token.lock:
            if self._token.value and self._token.expires_at > time.time() + 60:
                return self._token.value
            response = client.post(
                OAUTH_URL,
                headers={"Authorization": "Basic " + self._key, "RqUID": str(uuid4())},
                data={"scope": self._scope},
            )
            self._check_status(response)
            data = response.json()
            value, expiry = data.get("access_token"), data.get("expires_at")
            if (
                not isinstance(value, str)
                or not value
                or isinstance(expiry, bool)
                or not isinstance(expiry, (int, float))
            ):
                raise GigaChatError("GigaChat вернул некорректный токен доступа")
            expiry = expiry / 1000 if expiry > 100_000_000_000 else expiry
            if not math.isfinite(expiry) or expiry <= time.time():
                raise GigaChatError("GigaChat вернул истёкший токен доступа")
            self._token.value, self._token.expires_at = value, expiry
            return value

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        if not response.is_success:
            raise GigaChatError(
                f"Запрос к GigaChat не выполнен (HTTP {response.status_code})",
                response.status_code,
            )

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        with _request_slot(self._identity):
            refreshed = False
            for attempt in range(4):
                token = self._access_token(client)
                response = client.post(
                    EMBEDDINGS_URL,
                    headers={"Authorization": "Bearer " + token},
                    json={"model": self.model, "input": batch},
                )
                if response.status_code == 401 and not refreshed:
                    with self._token.lock:
                        if self._token.value == token:
                            self._token.value = ""
                    refreshed = True
                    continue
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    try:
                        delay = float(response.headers.get("Retry-After", ""))
                        if not math.isfinite(delay) or delay < 0:
                            raise ValueError
                    except ValueError:
                        delay = float(2**attempt)
                    # Keep the shared slot during cooldown, avoiding another burst.
                    time.sleep(min(delay, 10))
                    continue
                self._check_status(response)
                return self._vectors(response.json(), len(batch))
            self._check_status(response)
            raise GigaChatError("Не удалось обновить авторизацию GigaChat")

    def _vectors(self, data: Any, count: int) -> list[list[float]]:
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != count:
            raise GigaChatError("GigaChat вернул неполный набор эмбеддингов")
        vectors: dict[int, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise GigaChatError("GigaChat вернул некорректные эмбеддинги")
            index, values = row.get("index"), row.get("embedding")
            if (
                type(index) is not int
                or not 0 <= index < count
                or index in vectors
                or not isinstance(values, list)
                or len(values) != self.embedding_dim
            ):
                raise GigaChatError("GigaChat вернул некорректные эмбеддинги")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values
            ) or not any(values):
                raise GigaChatError("GigaChat вернул некорректный вектор")
            vectors[index] = values
        return [vectors[index] for index in range(count)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with httpx.Client(
                verify=tls_context(),
                timeout=60,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                result: list[list[float]] = []
                for start in range(0, len(texts), 16):
                    batch = texts[start : start + 16]
                    result.extend(self._embed_batch(client, batch))
                return result
        except GigaChatError:
            raise
        except Exception:
            raise GigaChatError("Ошибка соединения или ответа GigaChat") from None

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aclose(self) -> None:
        # Clients are scoped to a batch call; callbacks can run in different loops.
        pass


class GigaChatProvider(BaseLLMProvider):
    async def generate_with_usage(
        self, messages: Any, profile: Any = None, **kwargs: Any
    ) -> LLMResult:
        raise GigaChatError("GigaChat подключён только для эмбеддингов")

    def get_embeddings_model(self) -> GigaChatEmbedding:
        return GigaChatEmbedding()
