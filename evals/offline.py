"""Isolated integration fixtures, not a substitute for live model-quality evaluation."""

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from evals.evaluator import Runner


@contextmanager
def offline_pipeline() -> Iterator[Runner]:
    if os.getenv("PYTHON_DOTENV_DISABLED") != "1":
        raise ValueError("Offline mode requires PYTHON_DOTENV_DISABLED=1")
    # Import-time engines get inert local configuration, never inherited service URLs.
    with (
        patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql+asyncpg://eval:eval@127.0.0.1:1/eval_offline",
                "REDIS_URL": "redis://127.0.0.1:1/0",
            },
        ),
        _fixture_pipeline() as runner,
    ):
        yield runner


@contextmanager
def _fixture_pipeline() -> Iterator[Runner]:
    # Import only after rejecting accidental live configuration loading.
    from abbrev.expander import AbbrevExpander
    from evals.product import run_product
    from faq.matcher import FAQMatcher
    from llm.base import LLMResult
    from pipeline import llm_client

    class MemoryHistory:
        def __init__(self) -> None:
            self._storage: dict[str, list[dict]] = {}

        async def add_message(self, session_id: str, message: dict) -> None:
            self._storage.setdefault(session_id, []).append(message)

        async def get_history(self, session_id: str) -> list[dict]:
            return list(self._storage.get(session_id, []))

    mem_history = MemoryHistory()

    faq_items = [
        {"question": "Какие факультеты есть в НГУ?", "answer": "ФИТ, ФЕН, ГГФ."},
        {
            "question": "Сколько стоит общежитие в НГУ?",
            "answer": "Стоимость общежития уточняется при заселении.",
        },
        {
            "question": "Что означает НГУ (Новосибирский государственный университет)?",
            "answer": "НГУ — Новосибирский государственный университет.",
        },
    ]

    # Exact deterministic embeddings exercise the real FAQ cosine threshold and
    # input cleaning, without depending on a model or an untracked data file.
    class Embeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            phrases = [item["question"].casefold() for item in faq_items]
            return [
                [float(text.casefold() == phrase) for phrase in phrases]
                + [float(text.casefold() not in phrases)]
                for text in texts
            ]

    class FixtureProvider:
        async def generate_with_tools(
            self, messages, tools, tool_executor, profile=None, on_delta=None
        ) -> LLMResult:
            question = str(messages[-1].content)
            if "проходной балл" in question.casefold() and "2024" in question:
                if not any(tool.name == "get_admission_scores" for tool in tools):
                    raise AssertionError("Admission tool was not offered to provider")
                value = await tool_executor("get_admission_scores", {"year": 2024})
                return LLMResult(text=f"Проходной балл: {value}")
            if "добраться" in question.casefold() and any(
                "Тестовый контекст НГУ." in str(message.content) for message in messages
            ):
                return LLMResult(text="Тестовый контекст НГУ.")
            if "ммф" in question.casefold() or any(
                "Тестовый контекст ММФ." in str(message.content) for message in messages
            ):
                return LLMResult(text="Тестовый контекст ММФ.")
            return LLMResult(text="Информация не найдена.")

    async def history() -> MemoryHistory:
        return mem_history

    async def retrieve(query: str, conversation_history: str | None = None, **kwargs):
        if "ммф" in query.casefold():
            return "Тестовый контекст ММФ.", [{"url": "https://mmf.nsu.ru/admission"}]
        return "Тестовый контекст НГУ.", [{"url": "https://nsu.ru/info"}]

    async def config():
        return SimpleNamespace(enabled=True)

    async def threshold() -> float:
        return 0.99

    async def executor(name: str, args: dict) -> str:
        if name != "get_admission_scores":
            raise AssertionError("Unexpected tool")
        return "246 (тестовая фикстура, 2024)"

    def forbidden_db():
        raise AssertionError("Offline evaluation must not access PostgreSQL")

    with TemporaryDirectory(prefix="abitur-evals-") as directory, ExitStack() as stack:
        stack.enter_context(patch.object(AbbrevExpander, "_load", lambda self: None))
        expander = AbbrevExpander()
        # Deliberately only expand this case: other fixture questions keep their text.
        expander.load_items(
            [{"short": "НГУ", "full": "Новосибирский государственный университет"}]
        )
        # FAQ phrases go through the same expansion as user input.
        for item in faq_items[:2]:
            item["question"] = expander.expand(item["question"])
        matcher = FAQMatcher(
            faq_path=Path(directory) / "faq.yaml", threshold=0.99, embedder=Embeddings()
        )
        matcher.load_items(faq_items)
        replacements = {
            "get_redis_client": history,
            "get_abbrev_expander": lambda: expander,
            "get_faq_matcher": lambda: matcher,
            "get_llm_provider": FixtureProvider,
            "query_graph_with_sources": retrieve,
            "query_graph_with_crag": retrieve,
            "load_crag_config": config,
            "load_faq_threshold": threshold,
            "default_tool_executor": executor,
            "_spawn_bg": lambda coro: coro.close(),
            "AsyncSessionLocal": forbidden_db,
        }
        for name, value in replacements.items():
            stack.enter_context(patch.object(llm_client, name, value))
        yield run_product
