"""Шаги сборки ответа по отдельности — без БД, без поиска, без сети.

Смысл этих тестов в том, что они вообще возможны: раньше ask_local_llm была
одной функцией на 343 строки, и проверить «правильно ли собрался промпт» или
«что будет с пустым ответом модели» можно было только поднимая весь стек.
Именно поэтому evals ходили в query_graph в обход пайплайна (#305).
"""

import pytest

from llm.base import LLMResult, LLMUsage
from pipeline import llm_client
from pipeline.llm_client import (
    _build_messages,
    _expand_abbrevs,
    _postprocess,
    _run_tool_loop,
)


class TestPostprocess:
    def test_invented_link_keeps_only_text(self):
        """Ссылки, придуманные моделью, показывать нельзя — остаётся только текст."""
        result = _postprocess('Читай <a href="https://fake.test">тут</a>.', [])
        assert result.text == "Читай тут."

    def test_sources_pass_through(self):
        sources = [{"url": "https://nsu.ru/x", "title": "Приём"}]
        result = _postprocess("Ответ по существу.", sources)
        assert result.sources == sources

    @pytest.mark.parametrize(
        "refusal",
        [
            "Я не нашел информации об этом в базе знаний НГУ",
            "Я не нашёл информации об этом в базе знаний НГУ",
            "Информация не найдена в базе знаний",
        ],
    )
    def test_refusal_drops_sources(self, refusal):
        """К отказу список источников приклеивать бессмысленно."""
        result = _postprocess(refusal, [{"url": "https://nsu.ru/x", "title": "Т"}])
        assert result.sources == []
        assert result.text == refusal

    def test_empty_answer_gets_placeholder(self):
        result = _postprocess("   ", [{"url": "https://nsu.ru/x", "title": "Т"}])
        assert result.text == "Ответ не найден"
        assert result.sources == []

    def test_artifact_removed_from_answer_but_kept_in_log(self):
        """В лог пишем ответ до чистки: иначе не видно, что вернула модель."""
        result = _postprocess("Приём идёт foundland до 25 июля.", [])
        assert "foundland" not in result.text
        assert "foundland" in result.log_text


class TestBuildMessages:
    def test_context_goes_into_system_prompt(self):
        messages = _build_messages("Приём до 25 июля.", [])
        assert len(messages) == 1
        assert "Приём до 25 июля." in messages[0].content

    def test_history_roles_are_mapped(self):
        messages = _build_messages(
            "контекст",
            [
                {"role": "user", "content": "Когда подавать?"},
                {"role": "assistant", "content": "До 25 июля."},
                {"role": "system", "content": "это игнорируется"},
            ],
        )
        assert [m.type for m in messages] == ["system", "human", "ai"]
        assert messages[1].content == "Когда подавать?"
        assert messages[2].content == "До 25 июля."

    def test_sources_block_stripped_from_old_answers(self):
        """Иначе модель начинает копировать ссылки из своих прошлых ответов."""
        messages = _build_messages(
            "контекст",
            [
                {
                    "role": "assistant",
                    "content": (
                        "До 25 июля.\n\n<b>Источники:</b>\n"
                        '<a href="https://nsu.ru/x">Приём</a>'
                    ),
                }
            ],
        )
        assert messages[1].content == "До 25 июля."

    def test_inline_link_stripped_from_old_answers(self):
        messages = _build_messages(
            "контекст",
            [{"role": "assistant", "content": 'Смотри <a href="https://n.ru">тут</a>'}],
        )
        assert "<a" not in messages[1].content


class _FakeProvider:
    """Провайдер без сети: отдаёт заготовленный текст, при желании по частям."""

    def __init__(self, text: str, deltas: list[str] | None = None) -> None:
        self.text = text
        self.deltas = deltas or []
        self.tools_seen = None
        self.profile_seen = None

    async def generate_with_tools(
        self, messages, tools=None, tool_executor=None, profile=None, on_delta=None
    ):
        self.tools_seen = tools
        self.profile_seen = profile
        if on_delta is not None:
            for delta in self.deltas:
                await on_delta(delta)
        return LLMResult(text=self.text, usage=LLMUsage(total_tokens=42))


class TestRunToolLoop:
    @pytest.mark.asyncio
    async def test_returns_text_usage_and_provider(self, monkeypatch):
        provider = _FakeProvider("Ответ модели")
        monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)

        text, usage, name = await _run_tool_loop([], "session-1")

        assert text == "Ответ модели"
        assert usage.total_tokens == 42
        assert name == "_FakeProvider"

    @pytest.mark.asyncio
    async def test_admission_tool_is_offered(self, monkeypatch):
        """Инструмент проходных баллов должен быть доступен модели всегда."""
        provider = _FakeProvider("Ответ")
        monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)

        await _run_tool_loop([], "session-1")

        assert provider.tools_seen is not None
        assert [t.name for t in provider.tools_seen] == ["get_admission_scores"]

    @pytest.mark.asyncio
    async def test_streaming_reports_accumulated_text(self, monkeypatch):
        provider = _FakeProvider("Полный ответ", deltas=["Полный", " ответ"])
        monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)
        seen: list[str] = []

        async def on_chunk(text: str) -> None:
            seen.append(text)

        text, _, _ = await _run_tool_loop([], "session-1", on_chunk)

        assert seen == ["Полный", "Полный ответ"]
        assert text == "Полный ответ"

    @pytest.mark.asyncio
    async def test_broken_stream_callback_does_not_break_answer(self, monkeypatch):
        """Сбой отправки в канал не должен стоить пользователю ответа."""
        provider = _FakeProvider("Ответ модели", deltas=["Ответ", " модели"])
        monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)

        async def broken(text: str) -> None:
            raise RuntimeError("канал недоступен")

        text, _, _ = await _run_tool_loop([], "session-1", broken)

        assert text == "Ответ модели"

    @pytest.mark.asyncio
    async def test_empty_final_text_falls_back_to_streamed(self, monkeypatch):
        """Провайдер вернул пустой текст, но присылал дельты — берём накопленное."""
        provider = _FakeProvider("", deltas=["Собранный", " ответ"])
        monkeypatch.setattr(llm_client, "get_llm_provider", lambda: provider)

        text, _, _ = await _run_tool_loop([], "session-1", lambda _: _noop())

        assert text == "Собранный ответ"


async def _noop() -> None:
    return None


class TestExpandAbbrevs:
    def test_broken_expander_returns_original(self, monkeypatch):
        """Сбой расширителя аббревиатур не должен ронять ответ."""

        def boom():
            raise RuntimeError("нет данных аббревиатур")

        monkeypatch.setattr(llm_client, "get_abbrev_expander", boom)

        assert _expand_abbrevs("Что с ФИТ?", "session-1") == "Что с ФИТ?"
