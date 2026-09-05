import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, cast
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import wrap_embedding_func_with_attrs

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document
from db.postgres.services.document import ACTIVE_DOCUMENT_STATUSES
from llm.providers.gemini_graph_adapters import GeminiEmbedding, GeminiLLM
from llm.providers.openai_graph_adapters import OpenAIEmbedding, OpenAILLM
from rag import pg_storage

load_dotenv()

logger = logging.getLogger(__name__)

GRAPH_QUERY_MODES: frozenset = frozenset(
    ["naive", "local", "global", "hybrid", "mix", "bypass"]
)

DEFAULT_SOURCE_TITLE = "Источник информации"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
LOG_TEXT_PREVIEW_LIMIT = int(os.getenv("RAG_LOG_TEXT_PREVIEW_LIMIT", "2000"))
LOG_CHUNKS_LIMIT = int(os.getenv("RAG_LOG_CHUNKS_LIMIT", "10"))


def _clip_for_log(value: Any, limit: int = LOG_TEXT_PREVIEW_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... <truncated {len(text) - limit} chars>"


def _json_for_log(value: Any, limit: int = LOG_TEXT_PREVIEW_LIMIT) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return _clip_for_log(text, limit)


def _summarize_rag_chunk(chunk: dict, idx: int) -> dict[str, Any]:
    content = chunk.get("content") or chunk.get("text") or chunk.get("chunk_content")
    return {
        "index": idx,
        "file_path": chunk.get("file_path"),
        "source_id": chunk.get("source_id") or chunk.get("id"),
        "score": chunk.get("score") or chunk.get("distance"),
        "content": _clip_for_log(content, 700),
    }


def _log_rag_data_summary(
    *,
    graph_id: str,
    chunks: list,
    entities: list | None,
    relations: list | None,
) -> None:
    chunk_summaries = [
        _summarize_rag_chunk(chunk, idx)
        for idx, chunk in enumerate(chunks[:LOG_CHUNKS_LIMIT])
        if isinstance(chunk, dict)
    ]
    logger.info(
        (
            "RAG retrieval data: graph_id=%s chunks=%d entities=%d relations=%d "
            "logged_chunks=%d"
        ),
        graph_id,
        len(chunks),
        len(entities or []),
        len(relations or []),
        len(chunk_summaries),
    )
    logger.debug(
        "RAG retrieval chunks preview: graph_id=%s chunks=%s",
        graph_id,
        _json_for_log(chunk_summaries, limit=6000),
    )


def _clean_source_title(title: str | None) -> str | None:
    if not title:
        return None
    title = title.strip()
    if not title or title.startswith(("http://", "https://")):
        return None
    if UUID_RE.match(title):
        return None
    return title


def _fallback_title_from_source(source: str | None) -> str:
    if not source:
        return DEFAULT_SOURCE_TITLE

    source = source.strip()
    parsed = urlsplit(source)
    if parsed.path:
        filename = unquote(parsed.path.rstrip("/").split("/")[-1]).strip()
        if filename:
            return filename
    if parsed.netloc:
        return parsed.netloc

    clean_source = _clean_source_title(source)
    return clean_source or DEFAULT_SOURCE_TITLE


class _GraphWrapper:
    """Wrapper to track usage count of LightRAG instances for safe finalization."""

    def __init__(self, rag: LightRAG):
        self.rag = rag
        self.usage_count = 0
        self.marked_for_deletion = False


class GraphMemory:
    """
    Async manager for LightRAG graph memory.
    Uses OpenAI or Gemini for LLM and embeddings
    (controlled by LIGHTRAG_LLM_PROVIDER).
    """

    def __init__(self) -> None:
        self._graphs: Dict[str, _GraphWrapper] = {}
        self._graph_signatures: Dict[str, tuple[float, int]] = {}
        self._locks: Dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Lock] = {}
        self._last_disk_check: Dict[str, float] = {}
        self._clearing_events: Dict[str, asyncio.Event] = {}
        self._last_cache_clear_marker: Dict[str, float] = {}
        self._cache_clear_timeout: float = float(
            os.getenv("LIGHTRAG_CACHE_CLEAR_TIMEOUT", "10")
        )
        self._check_interval: float = 2.0  # limit disk scanning (os.scandir) frequency
        # Кэш «URL → заголовок» для блока «Источники»: строится из KV-файлов и
        # таблицы document, инвалидация — по mtime файлов и сигнатуре строк в БД.
        self._source_titles_cache: Dict[
            str, tuple[tuple[tuple[float, float], tuple[int, Any]], dict[str, str]]
        ] = {}

        self.workspace_path = os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag")
        os.makedirs(self.workspace_path, exist_ok=True)

        # LIGHTRAG_STORAGE=postgres: KV-стораджи и doc_status — в PostgreSQL
        # (JSONB), векторный индекс и граф остаются файловыми.
        self._pg_storage = pg_storage.is_pg_storage_enabled()
        if self._pg_storage:
            pg_storage.apply_postgres_env()

        logger.info(
            "GraphMemory initialized (async), workspace: %s, storage: %s",
            self.workspace_path,
            "postgres" if self._pg_storage else "json",
        )

    def _get_workspace_path(self, graph_id: str) -> str:
        path = os.path.join(self.workspace_path, graph_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_lock(self, graph_id: str) -> asyncio.Lock:
        """
        Get or create an asyncio.Lock scoped to the current event loop and graph_id.

        asyncio primitives are bound to the event loop in which they are created,
        so we must not share the same Lock instance across different loops.
        """
        loop = asyncio.get_running_loop()
        key = (loop, graph_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _get_cache_clear_marker_path(self, graph_id: str) -> str:
        workspace_path = self._get_workspace_path(graph_id)
        return os.path.join(workspace_path, "llm_cache_clear.marker")

    def _get_cache_clear_marker_mtime(self, graph_id: str) -> float:
        marker_path = self._get_cache_clear_marker_path(graph_id)
        try:
            with open(marker_path, "r", encoding="utf-8") as marker_file:
                raw_value = marker_file.read().strip()
                if raw_value:
                    return float(raw_value)
        except (OSError, ValueError):
            pass
        try:
            return os.path.getmtime(marker_path)
        except OSError:
            return 0.0

    async def _await_cache_clear_if_needed(self, graph_id: str) -> None:
        # Чтение marker-файла — синхронный disk I/O, уводим из event loop.
        marker_mtime = await asyncio.to_thread(
            self._get_cache_clear_marker_mtime, graph_id
        )
        last_seen = self._last_cache_clear_marker.get(graph_id, 0.0)

        if marker_mtime <= last_seen:
            return

        rag_to_clear = None

        start_time = time.monotonic()
        while True:
            async with self._get_lock(graph_id):
                wrapper = self._graphs.get(graph_id)
                if wrapper is None:
                    self._last_cache_clear_marker[graph_id] = marker_mtime
                    break

                if wrapper.usage_count <= 0:
                    rag_to_clear = wrapper.rag
                    self._last_cache_clear_marker[graph_id] = marker_mtime
                    break

            if time.monotonic() - start_time >= self._cache_clear_timeout:
                logger.warning(
                    f"Таймаут ожидания очистки кеша для графа {graph_id} (marker)"
                )
                self._last_cache_clear_marker[graph_id] = marker_mtime
                break

            await asyncio.sleep(0.05)

        if rag_to_clear is not None:
            try:
                await rag_to_clear.aclear_cache()
                logger.info(f"Очищен in-memory кеш LLM для графа: {graph_id} (marker)")
            except Exception as e:
                logger.warning(
                    f"Ошибка при очистке in-memory кеша {graph_id} по marker: {e}"
                )

        # Гарантируем, что на следующем запросе граф будет перечитан с диска,
        # иначе новый документ может не попасть в in-memory состояние.
        async with self._get_lock(graph_id):
            self._graph_signatures[graph_id] = (-1.0, -1)
            self._last_disk_check[graph_id] = 0.0

    def _get_workspace_signature(self, graph_id: str) -> tuple[float, int]:
        """Получает максимальное mtime и количество файлов для графа."""
        workspace_path = self._get_workspace_path(graph_id)

        latest_mtime: float = 0.0
        file_count: int = 0

        # Используем более лёгкую метрику, чтобы уменьшить нагрузку на диск:
        # - mtime директории workspace
        # - поверхностный обход содержимого через os.scandir без рекурсии
        # Обратите внимание: это по‑прежнему синхронный дисковый I/O и он может
        # блокировать event loop, просто делает это менее тяжело, чем рекурсивный обход.
        try:
            latest_mtime = os.path.getmtime(workspace_path)
            with os.scandir(workspace_path) as it:
                for entry in it:
                    if not entry.is_file() and not entry.is_dir():
                        continue
                    file_count += 1
                    try:
                        mtime = entry.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                    except OSError:
                        # Игнорируем проблемы с отдельными файлами/записями
                        continue
        except OSError:
            # Если не удаётся просканировать директорию, используем только её mtime
            pass
        return latest_mtime, file_count

    def _remove_doc_metadata(self, graph_id: str, doc_id: str) -> None:
        """Best-effort cleanup for LightRAG metadata files."""
        import json

        # В PG-режиме метаданные живут в БД: их удаляет сам LightRAG
        # (adelete_by_doc_id), файловой подчищать нечего.
        if self._pg_storage:
            return

        workspace_path = self._get_workspace_path(graph_id)
        metadata_files = [
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
        ]

        for filename in metadata_files:
            file_path = os.path.join(workspace_path, filename)
            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to read {file_path} for deletion: {exc}")
                continue

            if not isinstance(data, dict) or doc_id not in data:
                continue

            data.pop(doc_id, None)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"Failed to update {file_path} after deletion: {exc}")

    async def _get_or_create_graph(self, graph_id: str) -> _GraphWrapper:
        rag_to_finalize = None

        clearing_event = self._clearing_events.get(graph_id)
        if clearing_event is not None and not clearing_event.is_set():
            await clearing_event.wait()

        await self._await_cache_clear_if_needed(graph_id)

        async with self._get_lock(graph_id):
            now = time.monotonic()
            check_disk = False

            if graph_id not in self._graphs:
                check_disk = True
            elif now - self._last_disk_check.get(graph_id, 0.0) > self._check_interval:
                check_disk = True

            if check_disk:
                # os.scandir + stat — блокирующий I/O, выполняем в потоке.
                sig = await asyncio.to_thread(self._get_workspace_signature, graph_id)
                self._last_disk_check[graph_id] = now
                current_sig = self._graph_signatures.get(graph_id, (-1.0, -1))

                # Если граф уже загружен и файлы на диске не изменились — возвращаем кэш
                if graph_id in self._graphs and sig == current_sig:
                    wrapper = self._graphs[graph_id]
                    wrapper.usage_count += 1
                    return wrapper

                if graph_id in self._graphs:
                    logger.info(
                        f"Обнаружено обновление файлов графа {graph_id}."
                        + " Перезагрузка..."
                    )

                    old_wrapper = self._graphs.pop(graph_id, None)
                    if old_wrapper is not None:
                        old_wrapper.marked_for_deletion = True
                        if old_wrapper.usage_count <= 0:
                            rag_to_finalize = old_wrapper.rag
                        else:
                            logger.info(
                                f"Отложена финализация графа {graph_id}"
                                + f" (в использовании: {old_wrapper.usage_count})"
                            )
            elif graph_id in self._graphs:
                wrapper = self._graphs[graph_id]
                wrapper.usage_count += 1
                return wrapper

            # Если мы тут, значит графа нет в словаре. Создаем.
            workspace_path = self._get_workspace_path(graph_id)

            try:
                graph_llm_provider = os.getenv("LIGHTRAG_LLM_PROVIDER", "gemini").lower()

                if graph_llm_provider == "openai":
                    openai_api_key = os.getenv("OPENAI_API_KEY", "")
                    if not openai_api_key:
                        raise RuntimeError(
                            "LIGHTRAG_LLM_PROVIDER=openai,"
                            + " но OPENAI_API_KEY не установлен."
                        )
                    llm_adapter: Any = OpenAILLM(
                        api_key=openai_api_key,
                        model=os.getenv("OPENAI_GRAPH_MODEL", "gpt-4o-mini"),
                    )
                    embedding_adapter: Any = OpenAIEmbedding(
                        api_key=openai_api_key,
                        model=os.getenv(
                            "OPENAI_GRAPH_EMBEDDING_MODEL",
                            "text-embedding-3-small",
                        ),
                    )
                elif graph_llm_provider == "gemini":
                    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
                    if not gemini_api_key:
                        raise RuntimeError(
                            "LIGHTRAG_LLM_PROVIDER=gemini,"
                            + " но GEMINI_API_KEY не установлен."
                        )
                    llm_adapter: Any = GeminiLLM(
                        api_key=gemini_api_key,
                        model=os.getenv(
                            "GEMINI_GRAPH_MODEL", "gemini-3.1-flash-lite-preview"
                        ),
                    )
                    embedding_adapter: Any = GeminiEmbedding(
                        api_key=gemini_api_key,
                        model=os.getenv(
                            "GEMINI_GRAPH_EMBEDDING_MODEL",
                            "gemini-embedding-2-preview",
                        ),
                    )
                else:
                    raise RuntimeError(
                        f"LIGHTRAG_LLM_PROVIDER='{graph_llm_provider}' не поддерживается."
                        "Допустимые значения: openai, gemini"
                    )

                @wrap_embedding_func_with_attrs(
                    embedding_dim=embedding_adapter.embedding_dim
                )
                async def embedding_func(texts: List[str]) -> List[List[float]]:
                    return await embedding_adapter(texts)

                async def llm_model_func(prompt: str, **kwargs) -> str:
                    return await llm_adapter(prompt, **kwargs)

                embedding_workers = int(os.getenv("LIGHTRAG_EMBEDDING_WORKERS", "2"))
                llm_workers = int(os.getenv("LIGHTRAG_LLM_WORKERS", "1"))

                # PG-режим: KV и doc_status — в PostgreSQL (workspace=graph_id
                # разделяет базы знаний), векторный стор и граф — файловые.
                storage_kwargs: dict[str, Any] = {}
                if self._pg_storage:
                    await pg_storage.ensure_lightrag_tables()
                    storage_kwargs = {
                        "workspace": graph_id,
                        "kv_storage": "PGKVStorage",
                        "doc_status_storage": "PGDocStatusStorage",
                    }

                rag = LightRAG(
                    working_dir=workspace_path,
                    llm_model_func=llm_model_func,
                    embedding_func=embedding_func,
                    chunk_token_size=400,
                    chunk_overlap_token_size=50,
                    embedding_func_max_async=embedding_workers,
                    llm_model_max_async=llm_workers,
                    **storage_kwargs,
                )

                await rag.initialize_storages()
                await initialize_pipeline_status()

                wrapper = _GraphWrapper(rag)
                self._graphs[graph_id] = wrapper

                # Обновляем сигнатуру после инициализации,
                # чтобы не триггерить перезагрузку
                self._graph_signatures[graph_id] = await asyncio.to_thread(
                    self._get_workspace_signature, graph_id
                )
                self._last_disk_check[graph_id] = time.monotonic()
                logger.info(f"Created async LightRAG instance for graph: {graph_id}")

                # Инкрементируем использование сразу
                wrapper.usage_count += 1

            except Exception as e:
                logger.error(f"Error creating LightRAG instance for {graph_id}: {e}")
                raise

        # Вне лока финализируем старый граф, если он был удален и счетчик 0
        if rag_to_finalize is not None:
            try:
                await rag_to_finalize.finalize_storages()
            except Exception as finalize_err:
                logger.warning(
                    "Ошибка при финализации предыдущего " + "экземпляра LightRAG "
                    f"для графа {graph_id}: {finalize_err}"
                )

        return wrapper

    @asynccontextmanager
    async def _use_graph(self, graph_id: str):
        """Безопасно использует граф и декрементирует счетчик после
        завершения запроса."""
        # _get_or_create_graph теперь САМ инкрементирует usage_count под локом
        # перед тем как вернуть.
        # Это исключает race condition между получением и инкрементом.
        wrapper = await self._get_or_create_graph(graph_id)

        try:
            yield wrapper.rag
        finally:
            rag_to_finalize = None

            # Декремент и решение о финализации также выполняем под тем же локом
            async with self._get_lock(graph_id):
                wrapper.usage_count -= 1
                if wrapper.marked_for_deletion and wrapper.usage_count <= 0:
                    rag_to_finalize = wrapper.rag
            if rag_to_finalize is not None:
                try:
                    await rag_to_finalize.finalize_storages()
                    logger.info(f"Отложенная финализация графа завершена: {graph_id}")
                except Exception as e:
                    logger.warning(f"Ошибка при отложенной финализации: {e}")

    async def save(
        self,
        graph_id: str,
        text: str,
        source_id: Optional[str] = None,
        file_paths_str: Optional[str] = None,
    ) -> bool:
        try:
            async with self._use_graph(graph_id) as rag:
                await rag.ainsert(
                    text,
                    ids=source_id,
                    file_paths=file_paths_str or source_id,
                )
            # Обновляем кэш сигнатуры самого процесса, чтобы не было ложной инвалидации
            async with self._get_lock(graph_id):
                self._graph_signatures[graph_id] = await asyncio.to_thread(
                    self._get_workspace_signature, graph_id
                )
                self._last_disk_check[graph_id] = time.monotonic()
            return True
        except Exception as e:
            logger.error(f"Error inserting text into graph {graph_id}: {e}")
            return False

    async def query(
        self,
        graph_id: str,
        question: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
    ) -> str:
        try:
            async with self._use_graph(graph_id) as rag:
                logger.info(
                    "RAG query start: graph_id=%s mode=%s question=%s",
                    graph_id,
                    mode,
                    _clip_for_log(question),
                )
                result = await rag.aquery(question, param=QueryParam(mode=mode))
                logger.info(
                    "RAG query response: graph_id=%s mode=%s length=%d content=%s",
                    graph_id,
                    mode,
                    len(str(result)),
                    _clip_for_log(result),
                )
                return str(result)
        except Exception as e:
            logger.error(f"Error querying graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}"

    async def _retrieve_raw_chunks(
        self,
        rag: Any,
        question: str,
        mode: str,
    ) -> list[dict]:
        """Только ретрив: возвращает «сырые» чанки без генерации финального ответа.

        Используется CRAG-слоем, которому нужны кандидаты-чанки до генерации.
        """
        query_param = QueryParam(mode=cast(Any, mode))
        for attr_name, attr_value in (
            ("enable_rerank", True),
            ("chunk_top_k", 12),
            ("min_rerank_score", 0.0),
            ("only_need_context", True),
        ):
            if hasattr(query_param, attr_name):
                setattr(query_param, attr_name, attr_value)

        result = await rag.aquery_llm(question, param=query_param)
        data = result.get("data", {}) if isinstance(result, dict) else {}
        chunks = data.get("chunks", [])
        if not isinstance(chunks, list):
            return []
        return [c for c in chunks if isinstance(c, dict)]

    @staticmethod
    def _chunk_source_url(chunk: dict) -> str | None:
        file_path_value = str(chunk.get("file_path", "") or "").strip()
        for part in file_path_value.split(","):
            part = part.strip()
            if part.startswith("http://") or part.startswith("https://"):
                return part
        return None

    @staticmethod
    def _chunk_text(chunk: dict) -> str:
        text = (
            chunk.get("content") or chunk.get("text") or chunk.get("chunk_content") or ""
        )
        return str(text).strip()

    async def query_with_crag(
        self,
        graph_id: str,
        question: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
        max_sources: int = 3,
        conversation_history: str | None = None,
    ) -> tuple[str, list[dict]]:
        """CRAG-вариант ретрива: фильтрует чанки до генерации ответа.

        Возвращает (context, sources), где context — текст из чанков, прошедших
        CRAG-фильтр (грейдинг релевантности + авторитетная таблица факультетов),
        а sources — соответствующие URL. Финальную генерацию выполняет вызывающий
        код (llm_client) по этому контексту.
        """
        from rag.crag import CragChunk, load_crag_config, run_crag

        config = await load_crag_config()
        try:
            async with self._use_graph(graph_id) as rag:
                logger.info(
                    "CRAG retrieval start: graph_id=%s mode=%s question=%s",
                    graph_id,
                    mode,
                    _clip_for_log(question),
                )
                raw_chunks = await self._retrieve_raw_chunks(rag, question, mode)

                def _to_crag_chunks(items: list[dict]) -> list:
                    out = []
                    for idx, chunk in enumerate(items):
                        content = self._chunk_text(chunk)
                        if not content:
                            continue
                        out.append(
                            CragChunk(
                                index=idx,
                                content=content,
                                source_url=self._chunk_source_url(chunk),
                                file_path=str(chunk.get("file_path", "") or "") or None,
                            )
                        )
                    return out

                candidates = _to_crag_chunks(raw_chunks)

                async def _retrieve_fn(refined: str) -> list:
                    more = await self._retrieve_raw_chunks(rag, refined, mode)
                    return _to_crag_chunks(more)

                kept, hint = await run_crag(
                    question,
                    candidates,
                    retrieve_fn=_retrieve_fn,
                    config=config,
                )

                logger.info(
                    "CRAG retrieval result: graph_id=%s candidates=%d kept=%d "
                    "faculty=%s level=%s",
                    graph_id,
                    len(candidates),
                    len(kept),
                    hint.faculty.name if hint.faculty else None,
                    hint.level,
                )

                if not kept:
                    return "Релевантный контекст из базы знаний не найден.", []

                url_to_title = await self._get_source_titles(graph_id)

                context_parts: list[str] = []
                aggregated: dict[str, dict] = {}
                for chunk in kept:
                    context_parts.append(chunk.content)
                    url = chunk.source_url
                    if not url:
                        continue
                    current = aggregated.get(url)
                    if current is None:
                        aggregated[url] = {
                            "first_index": chunk.index,
                            "snippet": chunk.content,
                        }

                context = "\n\n---\n\n".join(context_parts)

                ordered = sorted(
                    aggregated.items(), key=lambda item: item[1]["first_index"]
                )
                sources: list[dict] = []
                for url, metrics in ordered[:max_sources]:
                    sources.append(
                        {
                            "url": url,
                            "title": url_to_title.get(url, DEFAULT_SOURCE_TITLE),
                        }
                    )

                return context, sources
        except Exception as e:
            logger.error(f"Error in CRAG query for graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}", []

    async def query_with_sources(
        self,
        graph_id: str,
        question: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
        max_sources: int = 3,
        conversation_history: str | None = None,
    ) -> tuple[str, list[dict]]:
        try:
            async with self._use_graph(graph_id) as rag:
                query_param = QueryParam(mode=mode)
                for attr_name, attr_value in (
                    ("enable_rerank", True),
                    ("chunk_top_k", 12),
                    ("min_rerank_score", 0.0),
                ):
                    if hasattr(query_param, attr_name):
                        setattr(query_param, attr_name, attr_value)

                query_settings = {
                    "mode": mode,
                    "enable_rerank": getattr(query_param, "enable_rerank", None),
                    "chunk_top_k": getattr(query_param, "chunk_top_k", None),
                    "top_k": getattr(query_param, "top_k", None),
                    "min_rerank_score": getattr(query_param, "min_rerank_score", None),
                }
                logger.info(
                    (
                        "RAG retrieval query: graph_id=%s settings=%s "
                        "history_present=%s question=%s"
                    ),
                    graph_id,
                    _json_for_log(query_settings),
                    bool(conversation_history),
                    _clip_for_log(question),
                )
                if conversation_history:
                    logger.debug(
                        "RAG retrieval conversation history: graph_id=%s history=%s",
                        graph_id,
                        _clip_for_log(conversation_history),
                    )

                result = await rag.aquery_llm(question, param=query_param)
                answer = result.get("llm_response", {}).get("content", "") or ""
                data = result.get("data", {})
                chunks = data.get("chunks", [])
                entities = data.get("entities")
                relations = data.get("relations")

                logger.info(
                    ("RAG retrieval response: graph_id=%s answer_length=%d answer=%s"),
                    graph_id,
                    len(str(answer)),
                    _clip_for_log(answer),
                )
                logger.debug(
                    "RAG retrieval raw response: graph_id=%s result=%s",
                    graph_id,
                    _json_for_log(result, limit=10000),
                )
                _log_rag_data_summary(
                    graph_id=graph_id,
                    chunks=chunks if isinstance(chunks, list) else [],
                    entities=entities if isinstance(entities, list) else None,
                    relations=relations if isinstance(relations, list) else None,
                )

                if not isinstance(chunks, list):
                    chunks = []

                url_to_title = await self._get_source_titles(graph_id)

                aggregated: dict[str, dict] = {}
                for idx, chunk in enumerate(chunks):
                    file_path_value = str(chunk.get("file_path", "") or "").strip()
                    if not file_path_value:
                        continue

                    chunk_text = (
                        chunk.get("content")
                        or chunk.get("text")
                        or chunk.get("chunk_content")
                        or ""
                    )
                    if chunk_text:
                        chunk_text = str(chunk_text).strip()

                    for source in file_path_value.split(","):
                        source = source.strip()
                        if not source.startswith("http://") and not source.startswith(
                            "https://"
                        ):
                            continue

                        current = aggregated.get(source)
                        if current is None:
                            aggregated[source] = {
                                "count": 1,
                                "first_index": idx,
                                "snippet": chunk_text,
                            }
                        else:
                            current["count"] += 1
                            if not current.get("snippet") and chunk_text:
                                current["snippet"] = chunk_text

                ordered_candidates = sorted(
                    aggregated.items(), key=lambda item: item[1]["first_index"]
                )
                sources_candidates = []
                for url, metrics in ordered_candidates:
                    title = url_to_title.get(url, "Источник информации")
                    snippet = str(metrics.get("snippet") or "").strip()
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "..."
                    sources_candidates.append(
                        {"url": url, "title": title, "snippet": snippet}
                    )

                reranked_sources = await self._rerank_sources_with_llm(
                    question=question,
                    conversation_history=conversation_history,
                    sources=sources_candidates,
                    max_sources=max_sources,
                )
                logger.info(
                    (
                        "RAG retrieval sources: graph_id=%s candidates=%d "
                        "selected=%d selected_sources=%s"
                    ),
                    graph_id,
                    len(sources_candidates),
                    len(reranked_sources),
                    _json_for_log(reranked_sources, limit=4000),
                )

                return str(answer), reranked_sources
        except Exception as e:
            logger.error(f"Error querying graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}", []

    @staticmethod
    def _docs_files_signature(workspace_path: str) -> tuple[float, float]:
        """mtime пары KV-файлов, по которым строятся заголовки источников."""

        def _mtime(name: str) -> float:
            try:
                return os.path.getmtime(os.path.join(workspace_path, name))
            except OSError:
                return 0.0

        return (
            _mtime("kv_store_doc_status.json"),
            _mtime("kv_store_full_docs.json"),
        )

    async def _documents_db_signature(self, graph_id: str) -> tuple[int, Any]:
        """Дешёвая сигнатура документов в БД: count + max(updated_at).

        Правка заголовка или смена статуса документа меняет updated_at
        (onupdate), поэтому пара однозначно определяет актуальность выборки.
        """
        from sqlalchemy import func, select

        try:
            async with AsyncSessionLocal() as session:
                stmt = select(
                    func.count(Document.id), func.max(Document.updated_at)
                ).where(
                    Document.graph_id == graph_id,
                    Document.status.in_(ACTIVE_DOCUMENT_STATUSES),
                )
                count, max_updated = (await session.execute(stmt)).one()
                return int(count or 0), max_updated
        except Exception as exc:
            logger.warning("Failed to load documents signature for sources: %s", exc)
            return -1, None

    async def _get_source_titles(self, graph_id: str) -> dict[str, str]:
        """Соответствие «URL → человекочитаемый заголовок» для «Источников».

        На каждом RAG-запросе проверяем только дешёвую сигнатуру (mtime двух
        KV-файлов + count/max(updated_at) из БД); полное чтение JSON и выборка
        документов выполняются лишь при её изменении. В PG-режиме роль mtime
        файлов играет сигнатура lightrag_doc_status (count + max(updated_at)).
        """
        if self._pg_storage:
            files_sig = await pg_storage.doc_status_signature(graph_id)
        else:
            workspace_path = await asyncio.to_thread(self._get_workspace_path, graph_id)
            files_sig = await asyncio.to_thread(
                self._docs_files_signature, workspace_path
            )
        db_sig = await self._documents_db_signature(graph_id)
        signature = (files_sig, db_sig)

        cached = self._source_titles_cache.get(graph_id)
        if cached is not None and cached[0] == signature:
            return cached[1]

        docs = await self.get_list_docs(graph_id)
        rag_doc_titles: dict[str, str] = {}
        url_to_title: dict[str, str] = {}

        for doc in docs:
            rag_doc_id = str(doc.get("id") or "").strip()
            doc_url = doc.get("url", "")
            if rag_doc_id:
                rag_doc_titles[rag_doc_id] = _fallback_title_from_source(
                    str(doc_url or rag_doc_id)
                )
            if doc_url:
                for part in str(doc_url).split(","):
                    url = part.strip()
                    if url:
                        url_to_title[url] = _fallback_title_from_source(url)

        try:
            from sqlalchemy import select

            async with AsyncSessionLocal() as session:
                stmt = select(Document).where(
                    Document.graph_id == graph_id,
                    Document.status.in_(ACTIVE_DOCUMENT_STATUSES),
                )
                result = await session.execute(stmt)
                for document in result.scalars().all():
                    title = _clean_source_title(document.title)
                    if not title:
                        title = _fallback_title_from_source(
                            document.source_url or document.rag_doc_id
                        )

                    rag_doc_titles[document.rag_doc_id] = title
                    if document.source_url:
                        url_to_title[document.source_url] = title
        except Exception as exc:
            logger.warning("Failed to load document titles for sources: %s", exc)

        for doc in docs:
            rag_doc_id = str(doc.get("id") or "").strip()
            doc_url = doc.get("url", "")
            title = rag_doc_titles.get(rag_doc_id)
            if not title or not doc_url:
                continue
            for part in str(doc_url).split(","):
                url = part.strip()
                if url:
                    url_to_title[url] = title

        self._source_titles_cache[graph_id] = (signature, url_to_title)
        return url_to_title

    async def _rerank_sources_with_llm(
        self,
        question: str,
        conversation_history: str | None,
        sources: list[dict],
        max_sources: int,
    ) -> list[dict]:
        if not sources:
            return []

        from llm.factory import get_llm_provider
        from llm.profiles import LLMProfiles

        history_block = conversation_history.strip() if conversation_history else ""
        sources_block = "\n".join(
            [
                f"- URL: {s['url']}\n  Title: {s['title']}\n  Snippet: {s['snippet']}"
                for s in sources
            ]
        )

        system_prompt = (
            "Ты — ассистент для ранжирования источников. "
            "Верни JSON-массив URL в порядке убывания релевантности. "
            "Используй только URL из списка источников. "
            "Никакого текста вокруг JSON."
        )

        user_parts = [f"Вопрос: {question}"]
        if history_block:
            user_parts.append(f"История переписки:\n{history_block}")
        user_parts.append(f"Источники:\n{sources_block}")
        user_prompt = "\n\n".join(user_parts)

        logger.info(
            "LLM rerank sources: candidates=%d, history_present=%s",
            len(sources),
            bool(history_block),
        )

        provider = get_llm_provider()
        raw = await provider.generate(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            profile=LLMProfiles.CHAT,
        )

        logger.debug("LLM rerank raw response: %s", raw)

        urls = None
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            try:
                urls = json.loads(match.group(0))
            except json.JSONDecodeError:
                urls = None

        if not isinstance(urls, list):
            logger.warning("LLM rerank fallback: invalid JSON list")
            return [{"url": s["url"], "title": s["title"]} for s in sources[:max_sources]]

        allowed = {s["url"]: s["title"] for s in sources}
        ordered = []
        for url in urls:
            if not isinstance(url, str):
                continue
            if url in allowed and url not in {item["url"] for item in ordered}:
                ordered.append({"url": url, "title": allowed[url]})

        if not ordered:
            logger.warning("LLM rerank fallback: no usable URLs")
            return [{"url": s["url"], "title": s["title"]} for s in sources[:max_sources]]

        logger.info("LLM rerank result: selected=%d", len(ordered[:max_sources]))
        return ordered[:max_sources]

    @staticmethod
    def _read_doc_list(workspace_path: str) -> list[dict]:
        """Синхронное чтение списка документов из KV-хранилищ.

        Выполняется через asyncio.to_thread: kv_store_*.json могут быть
        большими, их чтение не должно блокировать event loop.
        """
        status_file = os.path.join(workspace_path, "kv_store_doc_status.json")
        full_docs_file = os.path.join(workspace_path, "kv_store_full_docs.json")

        if not os.path.exists(status_file):
            return []

        full_docs_data = {}
        if os.path.exists(full_docs_file):
            try:
                with open(full_docs_file, "r", encoding="utf-8") as f:
                    full_docs_data = json.load(f)
            except Exception:
                pass

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = []
            for doc_id, info in data.items():
                url = (
                    info.get("url") or info.get("file_path") or info.get("file_paths_str")
                )
                if not url:
                    doc_full = full_docs_data.get(doc_id, {})
                    if isinstance(doc_full, dict):
                        # Сначала пытаемся из metadata или общих полей full_doc
                        # (в зависимости от формата памяти LightRAG
                        # это может быть metadata.get('file_path') и т.д.)
                        metadata = doc_full.get("metadata", {})
                        if isinstance(metadata, dict):
                            url = (
                                metadata.get("file_paths")
                                or metadata.get("file_path")
                                or metadata.get("url")
                            )

                        if not url:
                            url = (
                                doc_full.get("file_paths")
                                or doc_full.get("file_path")
                                or doc_full.get("url")
                            )

                if not url and str(doc_id).startswith("http"):
                    url = str(doc_id)

                result.append(
                    {
                        "id": doc_id,
                        "url": url,
                        "status": info.get("status"),
                        "content_summary": info.get("content_summary"),
                        "content_length": info.get("content_length"),
                        "created_at": info.get("created_at"),
                    }
                )

            # Сортируем документы: сначала новые (по created_at DESC)
            result.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            return result
        except Exception as e:
            logger.error(f"Error reading doc status for {workspace_path}: {e}")
            return []

    async def get_list_docs(self, graph_id: str) -> list[dict]:
        """Возвращает список документов из KV-хранилища статусов."""
        if self._pg_storage:
            return await pg_storage.list_docs(graph_id)
        workspace_path = await asyncio.to_thread(self._get_workspace_path, graph_id)
        return await asyncio.to_thread(self._read_doc_list, workspace_path)

    async def get_doc_full_text(self, graph_id: str, doc_id: str) -> Optional[str]:
        """Возвращает полный текст документа по его ID."""
        if self._pg_storage:
            return await pg_storage.get_full_doc(graph_id, doc_id)
        workspace_path = await asyncio.to_thread(self._get_workspace_path, graph_id)
        full_docs_file = os.path.join(workspace_path, "kv_store_full_docs.json")

        def _read_full_doc() -> Optional[str]:
            if not os.path.exists(full_docs_file):
                return None
            try:
                with open(full_docs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc_data = data.get(doc_id)
                if doc_data:
                    return doc_data.get("content")
                return None
            except Exception as e:
                logger.error(f"Error reading full docs for {doc_id} in {graph_id}: {e}")
                return None

        # kv_store_full_docs.json содержит все тексты целиком — читаем в потоке.
        return await asyncio.to_thread(_read_full_doc)

    async def get_doc_diagnostics(self, graph_id: str, doc_id: str) -> dict:
        """Диагностика документа по данным LightRAG.

        Возвращает статус обработки, число чанков, сущностей и связей,
        привязанных к документу. Сущностей/связей == 0 означает, что
        извлечение графа не дало результата и документ недостижим в режиме
        hybrid (нужен mix/naive).
        """
        import json

        if self._pg_storage:
            return await pg_storage.doc_diagnostics(graph_id, doc_id)

        workspace_path = self._get_workspace_path(graph_id)

        def _load(name: str) -> dict:
            path = os.path.join(workspace_path, name)
            if not os.path.exists(path):
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading {name} for {graph_id}: {e}")
                return {}

        status_data = _load("kv_store_doc_status.json")
        entities_data = _load("kv_store_full_entities.json")
        relations_data = _load("kv_store_full_relations.json")

        info = status_data.get(doc_id)
        if not isinstance(info, dict):
            return {
                "lightrag_status": "not_found",
                "chunks_count": 0,
                "entities_count": 0,
                "relations_count": 0,
                "content_length": None,
                "error": None,
            }

        metadata = info.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        error = (
            info.get("error")
            or info.get("error_msg")
            or metadata.get("error")
            or metadata.get("error_msg")
        )

        entities_entry = entities_data.get(doc_id)
        relations_entry = relations_data.get(doc_id)
        entities_count = (
            entities_entry.get("count") if isinstance(entities_entry, dict) else 0
        ) or 0
        relations_count = (
            relations_entry.get("count") if isinstance(relations_entry, dict) else 0
        ) or 0

        return {
            "lightrag_status": info.get("status") or "unknown",
            "chunks_count": info.get("chunks_count") or 0,
            "entities_count": entities_count,
            "relations_count": relations_count,
            "content_length": info.get("content_length"),
            "error": error,
        }

    async def debug_query(
        self,
        graph_id: str,
        question: str,
        mode: str = "hybrid",
        max_sources: int = 5,
    ) -> dict:
        """Выполняет запрос к базе знаний и возвращает «сырую» выдачу ретривала.

        Используется для диагностики: видно, какие чанки/сущности/связи нашёл
        конкретный режим, и попадает ли в выдачу нужный документ.
        """
        async with self._use_graph(graph_id) as rag:
            query_param = QueryParam(mode=cast(Any, mode))
            for attr_name, attr_value in (
                ("enable_rerank", True),
                ("chunk_top_k", 12),
                ("min_rerank_score", 0.0),
            ):
                if hasattr(query_param, attr_name):
                    setattr(query_param, attr_name, attr_value)

            settings = {
                "mode": mode,
                "enable_rerank": getattr(query_param, "enable_rerank", None),
                "chunk_top_k": getattr(query_param, "chunk_top_k", None),
                "top_k": getattr(query_param, "top_k", None),
                "min_rerank_score": getattr(query_param, "min_rerank_score", None),
            }

            result = await rag.aquery_llm(question, param=query_param)
            answer = result.get("llm_response", {}).get("content", "") or ""
            data = result.get("data", {})
            raw_chunks = data.get("chunks") or []
            raw_entities = data.get("entities") or []
            raw_relations = data.get("relations") or []
            if not isinstance(raw_chunks, list):
                raw_chunks = []

            url_to_title = await self._get_source_titles(graph_id)

            chunks: list[dict] = []
            for idx, chunk in enumerate(raw_chunks):
                if not isinstance(chunk, dict):
                    continue
                file_path_value = str(chunk.get("file_path", "") or "").strip()
                content = (
                    chunk.get("content")
                    or chunk.get("text")
                    or chunk.get("chunk_content")
                    or ""
                )
                source_url = None
                for part in file_path_value.split(","):
                    part = part.strip()
                    if part.startswith("http://") or part.startswith("https://"):
                        source_url = part
                        break
                score = chunk.get("rerank_score")
                if score is None:
                    score = chunk.get("score")
                chunks.append(
                    {
                        "index": idx,
                        "source_url": source_url,
                        "file_path": file_path_value or None,
                        "content": str(content).strip(),
                        "rerank_score": score,
                    }
                )

            entities: list[dict] = []
            if isinstance(raw_entities, list):
                for e in raw_entities:
                    if not isinstance(e, dict):
                        continue
                    entities.append(
                        {
                            "name": e.get("entity_name")
                            or e.get("name")
                            or e.get("entity")
                            or "",
                            "type": e.get("entity_type") or e.get("type"),
                            "description": e.get("description"),
                        }
                    )

            relations: list[dict] = []
            if isinstance(raw_relations, list):
                for r in raw_relations:
                    if not isinstance(r, dict):
                        continue
                    relations.append(
                        {
                            "source": r.get("src_id")
                            or r.get("source")
                            or r.get("src")
                            or r.get("entity1"),
                            "target": r.get("tgt_id")
                            or r.get("target")
                            or r.get("tgt")
                            or r.get("entity2"),
                            "description": r.get("description"),
                        }
                    )

            aggregated: dict[str, dict] = {}
            for c in chunks:
                url = c["source_url"]
                if not url:
                    continue
                current = aggregated.get(url)
                if current is None:
                    aggregated[url] = {
                        "count": 1,
                        "first_index": c["index"],
                        "snippet": c["content"],
                    }
                else:
                    current["count"] += 1
                    if not current.get("snippet") and c["content"]:
                        current["snippet"] = c["content"]

            ordered = sorted(aggregated.items(), key=lambda item: item[1]["first_index"])
            sources: list[dict] = []
            for url, metrics in ordered[:max_sources]:
                snippet = str(metrics.get("snippet") or "").strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                sources.append(
                    {
                        "url": url,
                        "title": url_to_title.get(url, "Источник информации"),
                        "snippet": snippet,
                    }
                )

            return {
                "question": question,
                "mode": mode,
                "answer": str(answer),
                "settings": settings,
                "chunks": chunks,
                "entities": entities,
                "relations": relations,
                "sources": sources,
            }

    async def export_stores(
        self, graph_id: str, include_embeddings: bool = False
    ) -> bytes:
        """Упаковывает сервисные сторы LightRAG в zip.

        По умолчанию исключает векторные базы (vdb_*.json): эмбеддинги
        привязаны к провайдеру (OpenAI на проде, Gemini локально) и
        непереносимы. Исходные тексты, статусы, чанки и граф включены.
        В PG-режиме KV-сторы дампятся из PostgreSQL в kv_store_*.json
        внутри архива (векторные базы и граф по-прежнему файловые).
        """
        import io
        import zipfile

        workspace_path = self._get_workspace_path(graph_id)
        skip_prefixes = () if include_embeddings else ("vdb_",)
        skip_files = {"llm_cache_clear.marker"}

        kv_dump: dict[str, str] = {}
        if self._pg_storage:
            kv_dump = await pg_storage.dump_kv_stores(graph_id)
            skip_files = set(skip_files) | set(kv_dump)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, payload in kv_dump.items():
                zf.writestr(os.path.join(graph_id, filename), payload)
            if os.path.isdir(workspace_path):
                for filename in sorted(os.listdir(workspace_path)):
                    file_path = os.path.join(workspace_path, filename)
                    if not os.path.isfile(file_path):
                        continue
                    if filename in skip_files:
                        continue
                    if any(filename.startswith(p) for p in skip_prefixes):
                        continue
                    zf.write(file_path, arcname=os.path.join(graph_id, filename))
        buffer.seek(0)
        return buffer.getvalue()

    async def delete_doc(self, graph_id: str, doc_id: str) -> bool:
        """Удаляет документ из RAG по его ID."""
        try:
            async with self._use_graph(graph_id) as rag:
                if hasattr(rag, "adelete_by_doc_id"):
                    # LightRAG удалит документ из векторной БД и графа
                    await rag.adelete_by_doc_id(doc_id)
                else:
                    logger.warning(
                        f"LightRAG не поддерживает"
                        f"adelete_by_doc_id (попытка удалить {doc_id})"
                    )
                    return False

            # Обновляем кэш сигнатуры
            async with self._get_lock(graph_id):
                # Перезапись KV-файлов и обход воркспейса — блокирующий I/O.
                await asyncio.to_thread(self._remove_doc_metadata, graph_id, doc_id)
                self._graph_signatures[graph_id] = await asyncio.to_thread(
                    self._get_workspace_signature, graph_id
                )
                self._last_disk_check[graph_id] = time.monotonic()

            await self.clear_cache(graph_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting doc {doc_id} from {graph_id}: {e}")
            return False

    async def cleanup(self, graph_id: Optional[str] = None) -> None:
        """
        Асинхронно финализировать хранилища для корректного завершения.

        Args:
            graph_id: ID конкретного графа или None для очистки всех графов.
        """
        try:
            if graph_id:
                async with self._get_lock(graph_id):
                    wrapper = self._graphs.pop(graph_id, None)
                    if wrapper is not None:
                        wrapper.marked_for_deletion = True
                        if wrapper.usage_count > 0:
                            logger.info(
                                f"Отложена финализация графа {graph_id} при cleanup"
                                + f" (в использовании: {wrapper.usage_count})"
                            )
                            wrapper = None

                if wrapper is not None:
                    await wrapper.rag.finalize_storages()
                    logger.info(
                        f"Хранилище финализировано и выгружено для графа: {graph_id}"
                    )
            else:
                # Если graph_id не указан, очищаем все графы в словаре.
                keys_to_clean = list(self._graphs.keys())
                for gid in keys_to_clean:
                    async with self._get_lock(gid):
                        wrapper = self._graphs.pop(gid, None)
                        if wrapper is not None:
                            wrapper.marked_for_deletion = True
                            if wrapper.usage_count > 0:
                                logger.info(
                                    f"Отложена финализация графа {gid} при cleanup"
                                    + f" (в использовании: {wrapper.usage_count})"
                                )
                                wrapper = None
                    if wrapper is not None:
                        await wrapper.rag.finalize_storages()
                        logger.info(
                            "Хранилище финализировано и" + f" выгружено для графа: {gid}"
                        )
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")

    async def clear_cache(self, graph_id: str) -> None:
        """
        Очищает кеш ответов LLM (kv_store_llm_response_cache.json)
        для указанного графа. Для этого очищаем in-memory кеш LightRAG,
        затем удаляем файл кеша на диске.
        """
        clearing_event = self._clearing_events.get(graph_id)
        if clearing_event is None or clearing_event.is_set():
            clearing_event = asyncio.Event()
            self._clearing_events[graph_id] = clearing_event

        try:
            rag_to_clear = None

            workspace_path = self._get_workspace_path(graph_id)
            marker_path = self._get_cache_clear_marker_path(graph_id)
            try:
                marker_value = str(time.time())
                with open(marker_path, "w", encoding="utf-8") as marker_file:
                    marker_file.write(marker_value)
                self._last_cache_clear_marker[graph_id] = float(marker_value)
            except Exception as e:
                logger.warning(
                    f"Не удалось обновить marker очистки кеша для {graph_id}: {e}"
                )

            start_time = time.monotonic()
            timed_out = False
            while True:
                async with self._get_lock(graph_id):
                    wrapper = self._graphs.get(graph_id)
                    if wrapper is None:
                        break
                    if wrapper.usage_count <= 0:
                        rag_to_clear = wrapper.rag
                        break

                if time.monotonic() - start_time >= self._cache_clear_timeout:
                    logger.warning(f"Таймаут ожидания очистки кеша для графа {graph_id}")
                    timed_out = True
                    break

                await asyncio.sleep(0.05)

            if timed_out:
                raise TimeoutError(
                    f"Cache clear timeout for graph {graph_id} after "
                    f"{self._cache_clear_timeout:.0f}s"
                )

            if rag_to_clear is not None:
                try:
                    await rag_to_clear.aclear_cache()
                    logger.info(f"Очищен in-memory кеш LLM для графа: {graph_id}")
                except Exception as e:
                    logger.warning(f"Ошибка при очистке in-memory кеша {graph_id}: {e}")

            cache_file = os.path.join(workspace_path, "kv_store_llm_response_cache.json")
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(
                        f"Очищен кеш ответов LLM для графа {graph_id}: "
                        f"удален файл {cache_file}"
                    )
                else:
                    logger.debug(
                        f"Файл кеша не найден, очистка не требуется: {cache_file}"
                    )
            except Exception as e:
                logger.error(f"Ошибка при удалении файла кеша {cache_file}: {e}")
        finally:
            clearing_event.set()


_graph_memory_instance = None


def get_graph_memory() -> GraphMemory:
    global _graph_memory_instance
    if _graph_memory_instance is None:
        _graph_memory_instance = GraphMemory()
    return _graph_memory_instance
