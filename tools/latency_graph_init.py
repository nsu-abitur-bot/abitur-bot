"""Process-local graph initialization subspans for explicitly invoked profiling.

No product defaults, caches, files or model calls are changed. Instrumentation
covers constructor/tokenizer/storage/metadata/client initialization independently.
"""

import hashlib
import json
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

from config.latency import stage


def instrument_graph_init() -> None:
    """Install diagnostic wrappers in this standalone process only."""
    import tiktoken
    import tiktoken.load
    from google import genai
    from lightrag import LightRAG
    from lightrag.lightrag import TiktokenTokenizer

    from llm.providers.gemini_graph_adapters import GeminiEmbedding
    from rag import graph_memory

    def sync_wrap(owner: Any, name: str, span: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with stage(span):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)

    def async_wrap(owner: Any, name: str, span: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            with stage(span):
                return await original(*args, **kwargs)

        setattr(owner, name, wrapped)

    sync_wrap(LightRAG, "__post_init__", "graph_constructor_inner")
    sync_wrap(TiktokenTokenizer, "__init__", "tokenizer_constructor")
    sync_wrap(tiktoken, "get_encoding", "graph_tokenizer_get_encoding")
    sync_wrap(tiktoken.load, "read_file", "tokenizer_source_read")
    sync_wrap(genai.Client, "__init__", "sdk_client_initialize")
    sync_wrap(GeminiEmbedding, "__init__", "graph_embedding_client_initialize")
    async_wrap(GeminiEmbedding, "aclose", "graph_embedding_client_close")
    async_wrap(LightRAG, "initialize_storages", "graph_storages_initialize_inner")
    async_wrap(
        graph_memory, "initialize_pipeline_status", "graph_pipeline_status_initialize"
    )
    async_wrap(
        graph_memory.GraphMemory,
        "_await_cache_clear_if_needed",
        "graph_cache_marker_wait",
    )
    sync_wrap(graph_memory.GraphMemory, "_get_workspace_signature", "graph_metadata_scan")

    original_read = tiktoken.load.read_file_cached

    @wraps(original_read)
    def cached_read(blobpath: str, *args: Any, **kwargs: Any) -> Any:
        directory = os.environ.get(
            "TIKTOKEN_CACHE_DIR",
            os.environ.get(
                "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")
            ),
        )
        filename = hashlib.sha1(blobpath.encode()).hexdigest()
        print(
            json.dumps(
                {
                    "event": "tokenizer_cache",
                    "enabled": bool(directory),
                    "file_present_before": bool(directory)
                    and (Path(directory) / filename).is_file(),
                }
            ),
            flush=True,
        )
        with stage("tokenizer_cache_read"):
            return original_read(blobpath, *args, **kwargs)

    tiktoken.load.read_file_cached = cached_read

    original_lock = graph_memory.GraphMemory._get_lock

    class ObservedLock:
        def __init__(self, lock: Any) -> None:
            self.lock = lock

        async def __aenter__(self) -> Any:
            with stage("graph_lock_wait"):
                return await self.lock.__aenter__()

        async def __aexit__(self, *args: Any) -> Any:
            return await self.lock.__aexit__(*args)

    def get_lock(self: Any, graph_id: str) -> Any:
        return ObservedLock(original_lock(self, graph_id))

    setattr(graph_memory.GraphMemory, "_get_lock", get_lock)
