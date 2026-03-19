import logging
import os
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

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

        self._graphs: Dict[str, LightRAG] = {}
        self._graph_last_mtime: Dict[str, float] = {}

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

    def _get_latest_mtime(self, graph_id: str) -> float:
        """Получает время последнего изменения хранилища (документов) для графа."""
        workspace_path = self._get_workspace_path(graph_id)

        latest_mtime: float = 0.0

        # Учитываем время изменения всех файлов в директории графа,
        # чтобы любые изменения workspace (graphml/kv_store_*.json и др.)
        # корректно инвалидацировали кэш.

        try:
            for root, _dirs, files in os.walk(workspace_path):
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(file_path)
                    except OSError:
                        # Игнорируем проблемы с отдельными файлами,
                        # продолжаем сканирование
                        continue
                    if mtime > latest_mtime:
                        latest_mtime = mtime
        except OSError:
            # Если директория недоступна, считаем, что данных нет
            return 0.0
        return latest_mtime

    async def _get_or_create_graph(self, graph_id: str) -> LightRAG:
        latest_mtime = self._get_latest_mtime(graph_id)
        current_mtime = self._graph_last_mtime.get(graph_id, -1.0)

        # Если граф уже загружен и файлы на диске не изменились — возвращаем кэш
        if graph_id in self._graphs and latest_mtime <= current_mtime:
            return self._graphs[graph_id]

        if graph_id in self._graphs:
            logger.info(
                f"Обнаружено обновление файлов графа {graph_id}. Перезагрузка..."
            )

            # Финализируем и удаляем старый экземпляр перед перезагрузкой,
            # чтобы избежать утечек ресурсов и неконсистентного состояния хранилищ.
            old_rag = self._graphs.pop(graph_id, None)
            if old_rag is not None:
                try:
                    await old_rag.finalize_storages()
                except Exception as finalize_err:
                    logger.warning(
                        "Ошибка при финализации предыдущего экземпляра LightRAG "
                        f"для графа {graph_id}: {finalize_err}"
                    )

        workspace_path = self._get_workspace_path(graph_id)

        try:
            graph_llm_provider = os.getenv("LIGHTRAG_LLM_PROVIDER", "gigachat").lower()

            if graph_llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                if not openai_api_key:
                    raise RuntimeError(
                        "LIGHTRAG_LLM_PROVIDER=openai, но OPENAI_API_KEY не установлен."
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
                        "LIGHTRAG_LLM_PROVIDER=gemini, но GEMINI_API_KEY не установлен."
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

            self._graphs[graph_id] = rag
            # Обновляем время загрузки, сохраняя фактическое максимальное mtime файлов
            self._graph_last_mtime[graph_id] = latest_mtime
            logger.info(f"Created async LightRAG instance for graph: {graph_id}")

        except Exception as e:
            logger.error(f"Error creating LightRAG instance for {graph_id}: {e}")
            raise

        return rag

    async def save(
        self,
        graph_id: str,
        text: str,
        source_id: Optional[str] = None,
        file_paths_str: Optional[str] = None,
    ) -> bool:
        try:
            rag = await self._get_or_create_graph(graph_id)
            await rag.ainsert(
                text,
                ids=source_id,
                file_paths=file_paths_str or source_id,
            )
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
            rag = await self._get_or_create_graph(graph_id)
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
            rag = await self._get_or_create_graph(graph_id)
            result = await rag.aquery_llm(question, param=QueryParam(mode=mode))
            answer = result.get("llm_response", {}).get("content", "") or ""
            references = result.get("data", {}).get("references", [])
            sources_set = set()
            for ref in references:
                for url in (ref.get("file_path", "") or "").split(","):
                    url = url.strip()
                    if url:
                        parsed = urlparse(url)
                        if parsed.scheme in ("http", "https"):
                            sources_set.add(url)
            return str(answer), list(sources_set)
        except Exception as e:
            logger.error(f"Error querying graph {graph_id}: {e}")
            return f"Error executing query: {str(e)}", []

    async def cleanup(self, graph_id: Optional[str] = None) -> None:
        """
        Асинхронно финализировать хранилища для корректного завершения.

        Args:
            graph_id: ID конкретного графа или None для очистки всех графов.
        """
        try:
            if graph_id and graph_id in self._graphs:
                await self._graphs[graph_id].finalize_storages()
                logger.info(f"Хранилище финализировано для графа: {graph_id}")
            elif not graph_id:
                for gid, rag in self._graphs.items():
                    await rag.finalize_storages()
                    logger.info(f"Хранилище финализировано для графа: {gid}")
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")


_graph_memory_instance = None


def get_graph_memory() -> GraphMemory:
    global _graph_memory_instance
    if _graph_memory_instance is None:
        _graph_memory_instance = GraphMemory()
    return _graph_memory_instance
