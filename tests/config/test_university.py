import hashlib

import pytest

from config import instance
from config.university import identity, is_nsu
from pipeline.prompts import SYSTEM_PROMPT_BASE


@pytest.fixture(autouse=True)
def restore_snapshot(monkeypatch):
    monkeypatch.setattr(instance, "_values", None)


def render() -> str:
    university = identity()
    return SYSTEM_PROMPT_BASE.format(
        context="",
        sources_hint="",
        assistant_name=university["assistant_name"],
        university_short_name=university["short_name"],
        university_full_name=university["full_name"],
    )


def test_nsu_prompt_is_unchanged():
    instance.install({}, 0)
    assert (
        hashlib.sha256(render().encode()).hexdigest()
        == "d4450e3e8571c2ca064d6400d14a5c9af4d93110160e68ce026a35d299d04a42"
    )
    assert is_nsu()


def test_custom_identity_and_prompt_contain_no_nsu():
    instance.install(
        {
            "UNIVERSITY_PROFILE": "custom",
            "UNIVERSITY_SHORT_NAME": "Тестовый вуз",
            "UNIVERSITY_FULL_NAME": "Тестовый университет",
        },
        1,
    )
    assert not is_nsu()
    assert "Тестовый университет" in render()
    assert "НГУ" not in render()
    assert "nsu.ru" not in str(identity())


@pytest.mark.asyncio
async def test_custom_disables_nsu_seeds_and_scheduler():
    from bootstrap.seed import seed_admission_scores, seed_reference_data
    from scheduler.main import run_sync_job

    instance.install(
        {
            "UNIVERSITY_PROFILE": "custom",
            "SEED_FACULTIES": "true",
            "SEED_ADMISSION_SCORES": "true",
            "SCHEDULER_ENABLED": "true",
        },
        1,
    )
    # These guards must return without even requiring a database or network.
    await seed_reference_data()
    await seed_admission_scores()
    result = await run_sync_job(None)
    assert not result.changed


def test_custom_does_not_read_legacy_abbreviations(tmp_path, monkeypatch):
    from abbrev import expander

    source = tmp_path / "abbrev.yaml"
    source.write_text("abbreviations:\n  - short: НГУ\n    full: чужие данные\n")
    monkeypatch.setattr(expander, "ABBREV_DATA_PATH", source)
    instance.install({"UNIVERSITY_PROFILE": "custom"}, 1)
    assert expander.AbbrevExpander().expand("НГУ") == "НГУ"
