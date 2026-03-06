import logging
import os
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import wrap_embedding_func_with_attrs

from llm.deepseek_graph_adapters import DeepSeekLLM
from llm.gigachat_graph_adapters import GigaChatEmbedding

load_dotenv()

logger = logging.getLogger(__name__)

GRAPH_QUERY_MODES: frozenset = frozenset(
    ["naive", "local", "global", "hybrid", "mix", "bypass"]
)


class GraphMemory:
    """
    Async manager for LightRAG graph memory.
    Uses DeepSeek for LLM and GigaChat for embeddings.
    """

    def __init__(
        self,
        credentials: Optional[str] = None,
        scope: str = "GIGACHAT_API_PERS",
        model_name: str = "deepseek-chat",
        embedding_model_name: str = "Embeddings",
    ) -> None:
        # GigaChat credentials for embeddings
        self.credentials: Optional[str] = credentials or os.getenv(
            "GIGACHAT_CREDENTIALS"
        )
        if not self.credentials:
            self.credentials = os.getenv("GIGACHAT_API_KEY")

        if not self.credentials:
            logger.warning(
                "GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY environment variable is not set."
            )

        self.scope: str = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model_name: str = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.embedding_model_name = embedding_model_name

        self._graphs: Dict[str, LightRAG] = {}

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

    async def _get_or_create_graph(self, graph_id: str) -> LightRAG:
        if graph_id in self._graphs:
            return self._graphs[graph_id]

        workspace_path = self._get_workspace_path(graph_id)

        try:
            llm_adapter = DeepSeekLLM(
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
            logger.info(f"Created async LightRAG instance for graph: {graph_id}")

        except Exception as e:
            logger.error(f"Error creating LightRAG instance for {graph_id}: {e}")
            raise

        return rag

    async def save(self, graph_id: str, text: str) -> bool:
        try:
            rag = await self._get_or_create_graph(graph_id)
            await rag.ainsert(text)
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
