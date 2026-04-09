import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import wrap_embedding_func_with_attrs

from llm.gemini_graph_adapters import GeminiEmbedding, GeminiLLM
from llm.gigachat_graph_adapters import GigaChatEmbedding, GigaChatLLM
from llm.openai_graph_adapters import OpenAIEmbedding, OpenAILLM

load_dotenv()

logger = logging.getLogger(__name__)

GRAPH_QUERY_MODES: frozenset = frozenset(
    ["naive", "local", "global", "hybrid", "mix", "bypass"]
)


class _GraphWrapper:
    """Wrapper to track usage count of LightRAG instances for safe finalization."""

    def __init__(self, rag: LightRAG):
        self.rag = rag
        self.usage_count = 0
        self.marked_for_deletion = False


class GraphMemory:
    """
    Async manager for LightRAG graph memory.
    Uses GigaChat or OpenAI for LLM and embeddings
    (controlled by LIGHTRAG_LLM_PROVIDER).
    """

    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
        model_name: str = "GigaChat",
        embedding_model_name: str = "Embeddings",
    ) -> None:
        # GigaChat credentials for LLM and embeddings
        self.credentials: Optional[str] = credentials or os.getenv(
            "GIGACHAT_CREDENTIALS"
        )
        if not self.credentials:
            self.credentials = os.getenv("GIGACHAT_API_KEY")

        if not self.credentials:
            logger.warning(
                "GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY "
                "environment variable is not set."
            )

        self.scope: str = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model_name: str = model_name or os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.embedding_model_name = embedding_model_name

        self._graphs: Dict[str, _GraphWrapper] = {}
        self._graph_signatures: Dict[str, tuple[float, int]] = {}
        self._locks: Dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Lock] = {}
        self._last_disk_check: Dict[str, float] = {}
        self._check_interval: float = 2.0  # limit disk scanning (os.scandir) frequency

        self.workspace_path = os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag")
        os.makedirs(self.workspace_path, exist_ok=True)

        logger.info(
            f"GraphMemory initialized (async) with model: {self.model_name}, "
            f"workspace: {self.workspace_path}"
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

    async def _get_or_create_graph(self, graph_id: str) -> _GraphWrapper:
        rag_to_finalize = None

        async with self._get_lock(graph_id):
            now = time.monotonic()
            check_disk = False

            if graph_id not in self._graphs:
                check_disk = True
            elif now - self._last_disk_check.get(graph_id, 0.0) > self._check_interval:
                check_disk = True

            if check_disk:
                sig = self._get_workspace_signature(graph_id)
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
                graph_llm_provider = os.getenv(
                    "LIGHTRAG_LLM_PROVIDER", "gigachat"
                ).lower()

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
                    llm_adapter = GigaChatLLM(
                        credentials=self.credentials,
                        scope=self.scope,
                        model=self.model_name,
                    )
                    embedding_adapter = GigaChatEmbedding(
                        credentials=self.credentials,
                        scope=self.scope,
                        model=self.embedding_model_name,
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

                rag = LightRAG(
                    working_dir=workspace_path,
                    llm_model_func=llm_model_func,
                    embedding_func=embedding_func,
                    chunk_token_size=400,
                    chunk_overlap_token_size=50,
                    embedding_func_max_async=embedding_workers,
                    llm_model_max_async=llm_workers,
                )

                await rag.initialize_storages()
                await initialize_pipeline_status()

                wrapper = _GraphWrapper(rag)
                self._graphs[graph_id] = wrapper

                # Обновляем сигнатуру после инициализации,
                # чтобы не триггерить перезагрузку
                self._graph_signatures[graph_id] = self._get_workspace_signature(
                    graph_id
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
                self._graph_signatures[graph_id] = self._get_workspace_signature(
                    graph_id
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
                result = await rag.aquery(question, param=QueryParam(mode=mode))
                return str(result)
        except Exception as e:
            logger.error(f"Error querying graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}"

    async def query_with_sources(
        self,
        graph_id: str,
        question: str,
        mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid",
    ) -> tuple[str, list[str]]:
        try:
            async with self._use_graph(graph_id) as rag:
                result = await rag.aquery_llm(question, param=QueryParam(mode=mode))
                answer = result.get("llm_response", {}).get("content", "") or ""
                references = result.get("data", {}).get("references", [])
                sources_set: set[str] = set()
                for ref in references:
                    file_paths_value = ref.get("file_path", "") or ""
                    for source in file_paths_value.split(","):
                        source = source.strip()
                        if source:
                            # Добавляем все непустые идентификаторы источников
                            # (как URL, так и локальные имена файлов).
                            sources_set.add(source)
                return str(answer), list(sources_set)
        except Exception as e:
            logger.error(f"Error querying graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}", []

    async def get_list_docs(self, graph_id: str) -> list[dict]:
        """Возвращает список документов из KV-хранилища статусов."""
        import json

        workspace_path = self._get_workspace_path(graph_id)
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
                url = info.get("url") or info.get("file_path") or info.get("file_paths_str")
                if not url:
                    doc_full = full_docs_data.get(doc_id, {})
                    if isinstance(doc_full, dict):
                        # Сначала пытаемся из metadata или общих полей full_doc
                        # (в зависимости от формата памяти LightRAG это может быть metadata.get('file_path') и т.д.)
                        metadata = doc_full.get("metadata", {})
                        if isinstance(metadata, dict):
                            url = metadata.get("file_paths") or metadata.get("file_path") or metadata.get("url")
                        
                        if not url:
                            url = doc_full.get("file_paths") or doc_full.get("file_path") or doc_full.get("url")
                            
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
            logger.error(f"Error reading doc status for {graph_id}: {e}")
            return []

    async def get_doc_full_text(self, graph_id: str, doc_id: str) -> Optional[str]:
        """Возвращает полный текст документа по его ID."""
        import json

        workspace_path = self._get_workspace_path(graph_id)
        full_docs_file = os.path.join(workspace_path, "kv_store_full_docs.json")

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
                            "Хранилище финализировано и"
                            + f" выгружено для графа: {gid}"
                        )
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")


_graph_memory_instance = None


def get_graph_memory() -> GraphMemory:
    global _graph_memory_instance
    if _graph_memory_instance is None:
        _graph_memory_instance = GraphMemory()
    return _graph_memory_instance
