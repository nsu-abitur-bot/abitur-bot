import asyncio

import pytest

from config import instance
from config.model_settings import materialize_models


def test_explicit_model_migration_unifies_routes_without_losing_unused_credentials():
    values = {
        "TEXT_PROVIDER": "deepseek",
        "TEXT_MODEL": "deepseek-flash",
        "EMBEDDING_PROVIDER": "gigachat",
        "EMBEDDING_MODEL": "EmbeddingsGigaR",
        "OPENAI_API_KEY": "unused-preserved",
        "PDF_PARSER_PROVIDER": "openai",
        "OPENAI_MODEL_VISION": "vision-model",
    }
    result = materialize_models(values)
    assert result["LLM_PROVIDER"] == result["LIGHTRAG_LLM_PROVIDER"] == "deepseek"
    assert result["DEEPSEEK_MODEL"] == result["DEEPSEEK_GRAPH_MODEL"] == "deepseek-flash"
    assert (
        result["FAQ_EMBEDDING_PROVIDER"]
        == result["LIGHTRAG_EMBEDDING_PROVIDER"]
        == "gigachat"
    )
    assert result["GIGACHAT_EMBEDDING_MODEL"] == "EmbeddingsGigaR"
    assert result["OPENAI_API_KEY"] == "unused-preserved"
    assert result["PDF_PARSER_PROVIDER"] == "openai"
    assert result["OPENAI_MODEL_VISION"] == "vision-model"


@pytest.mark.asyncio
async def test_inflight_question_keeps_a_consistent_settings_snapshot():
    instance.install({"LLM_PROVIDER": "gemini"}, 1)
    started, proceed = asyncio.Event(), asyncio.Event()

    @instance.pinned_settings()
    async def question():
        started.set()
        await proceed.wait()
        return instance.get_setting("LLM_PROVIDER"), instance.revision()

    task = asyncio.create_task(question())
    await started.wait()
    instance.install({"LLM_PROVIDER": "deepseek"}, 2)
    proceed.set()
    assert await task == ("gemini", 1)
    assert instance.get_setting("LLM_PROVIDER") == "deepseek"
    instance.install({}, 0)


def test_parallel_snapshot_factories_never_return_another_models_client(monkeypatch):
    import logging
    import time
    from concurrent.futures import ThreadPoolExecutor

    from config.instance import settings_scope
    from llm import factory

    class Provider:
        def __init__(self):
            self.model = instance.get_setting("DEEPSEEK_MODEL")

    monkeypatch.setattr("llm.providers.deepseek.DeepSeekProvider", Provider)
    monkeypatch.setattr(factory, "_provider_instance", None)
    monkeypatch.setattr(factory, "_provider_signature", ())

    # Logging releases the GIL while a client has already been installed.
    class Slow(logging.Handler):
        def emit(self, record):
            time.sleep(0.002)

    handler = Slow()
    logger = logging.getLogger("llm.factory")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)

    def request(number):
        model = "model-" + str(number)
        with settings_scope(
            {"LLM_PROVIDER": "deepseek", "DEEPSEEK_MODEL": model}, number
        ):
            assert getattr(factory.get_llm_provider(), "model") == model

    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(request, range(16)))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
