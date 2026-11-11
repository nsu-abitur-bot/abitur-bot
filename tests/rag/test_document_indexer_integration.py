import json
from typing import Any, cast

import pytest
from lightrag import QueryParam
from lightrag.kg.shared_storage import get_namespace_data

import rag.graph_memory as gm
from api.services.document_indexer import index_document
from db.postgres.models import Document
from db.postgres.services.document import DocumentService
from rag.graph_memory import GraphMemory


class FakeOpenAIEmbedding:
    embedding_dim = 16

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        # Детерминированный эмбеддинг фиксированной размерности
        return [
            [float(((hash(t) + i * 31) % 100) / 100.0) for i in range(self.embedding_dim)]
            for t in texts
        ]

    async def aclose(self) -> None:
        pass


class FakeOpenAILLM:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.extraction_response = "empty response<|COMPLETE|>"

    async def __call__(self, prompt: str, **kwargs: Any) -> str:
        if self.should_fail:
            raise RuntimeError("LLM extraction failed internally in LightRAG!")
        return self.extraction_response

    async def aclose(self) -> None:
        pass


class InMemoryDocument:
    def __init__(
        self,
        doc_id: str = "doc-1",
        rag_doc_id: str | None = None,
        title: str = "Правила приема",
        source_url: str = "https://nsu.ru/rules.pdf",
    ):
        self.id = doc_id
        self.rag_doc_id = rag_doc_id
        self.title = title
        self.source_url = source_url
        self.content_hash: str | None = None
        self.content_length: int | None = None
        self.status = "черновик"


class InMemoryDocumentService:
    def __init__(self, document: InMemoryDocument):
        self.document = document

    async def create_or_update_for_source(self, **kwargs: Any) -> InMemoryDocument:
        return self.document

    async def mark_indexed(self, doc_id: str, **kwargs: Any) -> InMemoryDocument:
        self.document.rag_doc_id = str(kwargs.get("rag_doc_id") or "")
        self.document.status = "проиндексирован"
        self.document.content_hash = kwargs.get("content_hash")
        self.document.content_length = kwargs.get("content_length")
        return self.document

    async def mark_error(self, doc_id: str) -> bool:
        self.document.status = "ошибка индексации"
        return True


@pytest.fixture(autouse=True)
async def clean_lightrag_pipeline_status():
    """Сбрасывает разделяемые структуры LightRAG перед и после каждого теста."""
    from lightrag.kg import shared_storage

    if shared_storage._shared_dicts is not None:
        shared_storage._shared_dicts.clear()
    if hasattr(shared_storage, "_init_flags") and shared_storage._init_flags is not None:
        shared_storage._init_flags.clear()
    yield
    if shared_storage._shared_dicts is not None:
        shared_storage._shared_dicts.clear()
    if hasattr(shared_storage, "_init_flags") and shared_storage._init_flags is not None:
        shared_storage._init_flags.clear()


@pytest.fixture
def real_lightrag_env(tmp_path, monkeypatch):
    """Изолированное окружение для реального LightRAG без внешних сетевых вызовов."""
    # Отключаем внешние сетевые вызовы через адаптеры
    fake_llm = FakeOpenAILLM()
    monkeypatch.setattr(gm, "OpenAIEmbedding", lambda **k: FakeOpenAIEmbedding())
    monkeypatch.setattr(gm, "OpenAILLM", lambda **k: fake_llm)
    monkeypatch.setenv("LIGHTRAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-external-calls")

    memory = GraphMemory()
    memory.workspace_path = str(tmp_path)
    monkeypatch.setattr("api.services.document_indexer.get_graph_memory", lambda: memory)
    monkeypatch.setattr("rag.loader.get_graph_memory", lambda: memory)

    return memory, fake_llm


@pytest.mark.asyncio
async def test_real_lightrag_failed_status_preserves_old_version(
    real_lightrag_env,
):
    """[P1] Внутренний сбой extraction в LightRAG без исключения из ainsert().

    Старая версия и её идентификатор в Postgres должны остаться нетронутыми.
    """
    memory, fake_llm = real_lightrag_env
    graph_id = "test_graph_fail"

    # 1. Успешно индексируем первую версию
    doc = InMemoryDocument(doc_id="doc-rules", rag_doc_id=None)
    service = InMemoryDocumentService(doc)

    doc_v1 = await index_document(
        text="Первая рабочая версия правил приёма в НГУ.",
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v1 is not None
    old_rag_doc_id = doc_v1.rag_doc_id
    assert old_rag_doc_id is not None
    assert doc.status == "проиндексирован"

    # Проверяем, что первая версия в LightRAG имеет статус PROCESSED
    status_v1 = await memory.get_doc_status(graph_id, old_rag_doc_id)
    assert status_v1 == "processed"

    # 2. Включаем контролируемый сбой LLM для второй версии
    fake_llm.should_fail = True

    # 3. Индексируем новую версию документа
    doc_v2 = await index_document(
        text="Вторая версия, при индексации которой падает LLM extraction.",
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=2.0,
    )

    # 4. Проверяем блокировку переключения
    assert doc_v2 is None
    # Статус в БД стал ошибкой, но rag_doc_id НЕ переключился на сломанную версию!
    assert doc.status == "ошибка индексации"
    assert doc.rag_doc_id == old_rag_doc_id

    # Старая версия в LightRAG по-прежнему цела
    status_v1_after = await memory.get_doc_status(graph_id, old_rag_doc_id)
    assert status_v1_after == "processed"


@pytest.mark.asyncio
async def test_real_lightrag_busy_queue_preserves_old_version(
    real_lightrag_env,
):
    """[P1] При занятой очереди (pipeline_status['busy']=True) задача остаётся PENDING.

    ainsert() возвращается сразу, но статус не PROCESSED. Переключение и удаление
    старой версии должны быть заблокированы по таймауту ожидания.
    """
    memory, fake_llm = real_lightrag_env
    graph_id = "test_graph_busy"

    # 1. Успешно индексируем первую версию
    doc = InMemoryDocument(doc_id="doc-busy", rag_doc_id=None)
    service = InMemoryDocumentService(doc)

    doc_v1 = await index_document(
        text="Действующая версия документа до возникновения очереди.",
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v1 is not None
    old_rag_doc_id = doc_v1.rag_doc_id
    assert old_rag_doc_id is not None

    # 2. Симулируем занятый пайплайн в LightRAG
    async with memory._use_graph(graph_id):
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status["busy"] = True

    try:
        # 3. Попытка обновления при занятой очереди с коротким таймаутом
        doc_v2 = await index_document(
            text="Новая версия текста, отправленная в занятую очередь.",
            document=cast(Document, doc),
            graph_id=graph_id,
            document_service=cast(DocumentService, service),
            wait_timeout=0.05,
        )

        # 4. Переключение не произошло, старая версия сохранена
        assert doc_v2 is None
        assert doc.status == "ошибка индексации"
        assert doc.rag_doc_id == old_rag_doc_id

        # Старая версия в LightRAG не удалена
        status_v1 = await memory.get_doc_status(graph_id, old_rag_doc_id)
        assert status_v1 == "processed"
    finally:
        async with memory._use_graph(graph_id):
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status["busy"] = False


@pytest.mark.asyncio
async def test_real_lightrag_reupload_identical_text_findable_after_delete(
    real_lightrag_env,
):
    """[P1] Повторная загрузка идентичного текста не удаляет чанки новой версии.

    После add → switch → delete старая версия удаляется, но новая версия
    должна полностью находиться поиском (общие чанки сохранены).
    """
    memory, fake_llm = real_lightrag_env
    graph_id = "test_graph_identical"

    text = (
        "Новосибирский государственный университет проводит приём на факультет"
        " информационных технологий в 2026 году. Минимальный балл составляет 270."
    )

    doc = InMemoryDocument(doc_id="doc-identical", rag_doc_id=None)
    service = InMemoryDocumentService(doc)

    # 1. Первая загрузка
    doc_v1 = await index_document(
        text=text,
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v1 is not None
    old_rag_doc_id = doc_v1.rag_doc_id
    assert old_rag_doc_id is not None

    # 2. Повторная загрузка абсолютно идентичного текста (переиндексация)
    doc_v2 = await index_document(
        text=text,
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v2 is not None
    new_rag_doc_id = doc_v2.rag_doc_id
    assert new_rag_doc_id != old_rag_doc_id

    # 3. Старый документ удалён из doc_status
    old_status = await memory.get_doc_status(graph_id, old_rag_doc_id)
    assert old_status is None

    # Новый документ в статусе processed
    new_status = await memory.get_doc_status(graph_id, new_rag_doc_id)
    assert new_status == "processed"

    # 4. Проверяем, что текст полностью находится поиском через LightRAG
    async with memory._use_graph(graph_id) as rag:
        query_res = await rag.aquery_llm(
            "Минимальный балл на факультет информационных технологий",
            param=QueryParam(mode="naive"),
        )
        assert query_res.get("status") == "success"
        chunks = (query_res.get("data") or {}).get("chunks") or []
        assert len(chunks) > 0
        assert any(
            "Новосибирский государственный университет" in c["content"] for c in chunks
        )


@pytest.mark.asyncio
async def test_real_lightrag_partial_update_with_shared_chunks_findable_after_delete(
    real_lightrag_env,
):
    """[P1] Частичное обновление: документ содержит и общие, и изменившиеся части.

    После удаления старой версии:
    - Общие чанки и новые чанки остаются доступными в поиске.
    - Устаревшие уникальные чанки удаляются.
    """
    memory, fake_llm = real_lightrag_env
    graph_id = "test_graph_partial"

    # Формируем текст из двух явно выраженных частей достаточной длины,
    # чтобы образовались отдельные чанки (при chunk_token_size=400).
    shared_part = (
        "Общая информация для абитуриентов Новосибирского государственного университета: "
        "приём документов начинается 20 июня 2026 года. Документы можно подать онлайн "
        "через личный кабинет абитуриента НГУ с верификацией через Госуслуги. "
    ) * 15
    old_exclusive_part = (
        "Старый регламент заселения в студенческий городок НГУ: заселение производится "
        "строго по бумажным ордерам с 25 августа в порядке живой очереди в дирекции. "
        "Обязательно предоставить оригинал медицинской справки формы 086-у. "
    ) * 15
    new_exclusive_part = (
        "Новый регламент заселения в студгородок НГУ: заселение производится "
        "онлайн с выбором комнаты в приложении. "
        "Медицинская справка загружается в виде скана. "
    ) * 15

    doc = InMemoryDocument(doc_id="doc-partial", rag_doc_id=None)
    service = InMemoryDocumentService(doc)

    # 1. Первая версия: shared_part + old_exclusive_part
    text_v1 = f"{shared_part}\n\n{old_exclusive_part}"
    doc_v1 = await index_document(
        text=text_v1,
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v1 is not None
    old_rag_doc_id = doc_v1.rag_doc_id
    chunks_v1 = await memory.get_chunks_list(graph_id, old_rag_doc_id)
    assert len(chunks_v1) >= 2

    # 2. Вторая версия: shared_part + new_exclusive_part
    text_v2 = f"{shared_part}\n\n{new_exclusive_part}"
    doc_v2 = await index_document(
        text=text_v2,
        document=cast(Document, doc),
        graph_id=graph_id,
        document_service=cast(DocumentService, service),
        wait_timeout=5.0,
    )
    assert doc_v2 is not None
    new_rag_doc_id = doc_v2.rag_doc_id
    chunks_v2 = await memory.get_chunks_list(graph_id, new_rag_doc_id)
    assert len(chunks_v2) >= 2

    # Проверяем, что есть действительно общий чанк
    shared_chunk_ids = set(chunks_v1) & set(chunks_v2)
    assert len(shared_chunk_ids) > 0

    # 3. Поиск по общему фрагменту должен успешно находить чанк
    async with memory._use_graph(graph_id) as rag:
        query_shared = await rag.aquery_llm(
            "Приём документов начинается 20 июня личный кабинет",
            param=QueryParam(mode="naive"),
        )
        assert query_shared.get("status") == "success"
        shared_found = (query_shared.get("data") or {}).get("chunks") or []
        assert len(shared_found) > 0
        assert any(
            "личный кабинет абитуриента" in c.get("content", "") for c in shared_found
        )

        # Проверяем, что общие чанки присутствуют в хранилище text_chunks
        shared_in_storage = await rag.text_chunks.get_by_ids(list(shared_chunk_ids))
        assert all(c is not None for c in shared_in_storage)

        # Поиск по новому фрагменту находит новый чанк
        query_new = await rag.aquery_llm(
            "выбором комнаты в приложении",
            param=QueryParam(mode="naive"),
        )
        assert query_new.get("status") == "success"
        new_found = (query_new.get("data") or {}).get("chunks") or []
        assert len(new_found) > 0
        assert any("выбором комнаты" in c.get("content", "").lower() for c in new_found)

        # Устаревшие уникальные чанки из v1 больше не находятся в text_chunks
        old_exclusive_chunks = set(chunks_v1) - set(chunks_v2)
        assert len(old_exclusive_chunks) > 0
        in_text_chunks = await rag.text_chunks.get_by_ids(list(old_exclusive_chunks))
        assert all(c is None for c in in_text_chunks)


@pytest.mark.asyncio
async def test_program_exam_cards_from_existing_index(real_lightrag_env):
    memory, _ = real_lightrag_env
    graph_id = "test_graph_program_exams"
    base_url = "https://priem.sibsiu.ru/specialities/"
    exam_text = (
        "## Экзамены для поступающих\n"
        "##### Единый государственный экзамен (ЕГЭ)\n"
        "###### Обязательные предметы\n"
        "- Математика (профильная) (мин. балл 40)\n"
        "- Русский язык (мин. балл 40)\n"
        "###### Предметы на выбор\n"
        "- Один на выбор: физика (41), информатика и ИКТ (46), "
        "география (40) или химия (40).\n"
        "##### Для поступающих после среднего профессионального образования\n"
        "- Физические основы технических наук (мин. балл 55)\n"
    )
    cards = {
        "322": (
            "# Прикладная геология\nКод специальности: 21.05.02\n#### Специалитет\n\n"
            + "Подробности об учебной программе и практических занятиях. " * 80
            + "\n\n"
            + exam_text
            + "\n## 09.03.01 Информатика и вычислительная техника\n"
            "### ЕГЭ\n- Математика (профильная) — 99\n"
        ),
        "101": (
            "# 09.03.01 Информатика и вычислительная техника\n"
            "Экзамены для поступающихЕдиный государственный экзамен (ЕГЭ)"
            "Математика (профильная) — 70"
            "Для поступающих после среднего профессионального образования"
            "Физические основы — 99"
        ),
        "102": (
            "# 09.03.02 Информационные системы и технологии\n"
            "## Стоимость обучения\n200 000 рублей\n"
        ),
        "103": (
            "# 09.03.03 Прикладная информатика — Цифровая экономика\n"
            "## ЕГЭ\n- Математика (профильная) — 50\n"
        ),
        "104": (
            "# 09.03.03 Прикладная информатика — Аналитика данных\n"
            "## ЕГЭ\n- Физика — 60\n"
        ),
    }
    try:
        async with memory._use_graph(graph_id) as rag:
            rag.chunk_token_size = 70
            rag.chunk_overlap_token_size = 18
        for card_id, text in cards.items():
            assert await memory.save(
                graph_id,
                text,
                file_paths_str=base_url + card_id,
                source_id="card-" + card_id,
            )
        await memory.cleanup(graph_id)
        for question in (
            "Какие ЕГЭ нужны на 21.05.02?",
            "Какие минимальные баллы ЕГЭ нужны на 21.05.02?",
            "Какие ЕГЭ нужны на 21.05.02 специалитет?",
            "«ПРИКЛАДНАЯ   ГЕОЛОГИЯ» специалитет какие ЕГЭ нужны?",
        ):
            result = await memory.query_with_crag(graph_id, question)
            assert "Математика (профильная) (мин. балл 40)" in result.context
            assert "Русский язык (мин. балл 40)" in result.context
            assert "физика (41), информатика и ИКТ (46)" in result.context
            assert "география (40) или химия (40)" in result.context
            assert "Физические основы технических наук" not in result.context
            assert "99" not in result.context
            assert "Уровень образования — специалитет" in result.context
            assert "специалитет" in (result.generation_question or "")
            assert [source["url"] for source in result.sources] == [base_url + "322"]
            assert any("card-322" in line for line in result.trace)
            reason = "code_mismatch" if "21.05.02" in question else "name_mismatch"
            assert any("card-101" in line and reason in line for line in result.trace)
            assert (
                sum("card-322" in line and " kept " in line for line in result.trace) >= 2
            )

        result = await memory.query_with_sources(graph_id, "Какие ЕГЭ нужны на 21.05.02?")
        assert [source["url"] for source in result.sources] == [base_url + "322"]

        result = await memory.query_with_crag(graph_id, "Какие ЕГЭ нужны на 09.03.01?")
        assert "70" in result.context
        assert "Физические основы" not in result.context
        assert "99" not in result.context
        assert [source["url"] for source in result.sources] == [base_url + "101"]

        result = await memory.query_with_crag(graph_id, "Какие ЕГЭ нужны на 09.03.02?")
        assert result.answer and "нет данных о ЕГЭ" in result.answer
        assert not result.context and not result.sources
        result = await memory.query_with_crag(
            graph_id, "Какие ЕГЭ нужны на Прикладная геология, бакалавриат?"
        )
        assert result.answer and "нет данных о ЕГЭ" in result.answer
        assert not result.context and not result.sources

        result = await memory.query_with_crag(graph_id, "Какие ЕГЭ нужны на 09.03.03?")
        assert result.answer and "Уточните профиль" in result.answer
        assert (
            "Цифровая экономика" in result.answer and "Аналитика данных" in result.answer
        )
        assert not result.context and not result.sources
        history = (
            f"user: Какие ЕГЭ нужны на 09.03.03?\nassistant: {result.answer}\n"
            "user: Аналитика данных"
        )
        follow_up = await memory.query_with_crag(
            graph_id, "Аналитика данных", conversation_history=history
        )
        assert "Физика — 60" in follow_up.context and "50" not in follow_up.context
        assert [source["url"] for source in follow_up.sources] == [base_url + "104"]
        assert follow_up.generation_question
        assert "Какие ЕГЭ нужны на 09.03.03?" in follow_up.generation_question

        result = await memory.query_with_crag(
            graph_id, "Какие ЕГЭ нужны на 09.03.03, профиль «Аналитика данных»?"
        )
        assert "Физика — 60" in result.context and "50" not in result.context
        assert [source["url"] for source in result.sources] == [base_url + "104"]

        result = await memory.query_with_crag(
            graph_id, "Какие ЕГЭ нужны на 09.03.03, профиль «Нейросети»?"
        )
        assert result.answer and "нет данных о ЕГЭ" in result.answer
        assert not result.context and not result.sources
        assert any("profile_mismatch" in line for line in result.trace)
    finally:
        await memory.cleanup(graph_id)


async def test_program_exam_card_preserves_scores_from_history(
    real_lightrag_env, monkeypatch
):
    from llm.base import LLMResult
    from pipeline import llm_client

    memory, _ = real_lightrag_env
    graph_id = "test_program_exam_history"
    question = "Какие ЕГЭ нужны на 09.03.01, с моими баллами могу поступить?"
    previous = "У меня 44 по информатике и 70 по математике"
    history = [{"role": "user", "content": previous}]

    class Provider:
        async def generate_with_tools(self, messages, **kwargs):
            assert "Информатика — 46" in str(messages[0].content)
            assert [
                message.content for message in messages if message.type == "human"
            ] == [
                previous,
                f"{question}\nУровень образования — бакалавриат.",
            ]
            return LLMResult(text="44 по информатике ниже минимального балла 46.")

        async def generate(self, messages, profile) -> str:
            payload = json.loads(messages[-1].content)
            assert payload["history"] == [{"role": "human", "content": previous}]
            assert any("Информатика — 46" in source for source in payload["sources"])
            return '{"verdict":"supported","quotes":["Информатика — 46"]}'

    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)
    try:
        assert await memory.save(
            graph_id,
            "# 09.03.01 Информатика и вычислительная техника\n"
            "## ЕГЭ\n- Информатика — 46\n- Математика (профильная) — 40\n",
            file_paths_str="https://example.org/program",
            source_id="card-101",
        )
        retrieval = await memory.query_with_crag(
            graph_id, question, conversation_history=f"user: {previous}\nuser: {question}"
        )
        rag = llm_client._RagResult(
            context=retrieval.context,
            generation_question=retrieval.generation_question,
            evidence=retrieval.evidence,
        )
        ctx = llm_client._Ctx(session_id="ege-history", user_id=0)
        answer, _ = await llm_client._generate_answer(rag, history, question, ctx)
        assert answer == "44 по информатике ниже минимального балла 46."
    finally:
        await memory.cleanup(graph_id)
