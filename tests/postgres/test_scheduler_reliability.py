"""Надёжность планировщика конкурсных списков.

#285 — уведомления не теряются при сбое посреди обхода: они лежат в
notification_outbox, записанные в одном коммите с новыми местами.
#286 — ошибка загрузки списка не выглядит как пустой список: видна в логе
как ERROR и не обновляет хэш упавшего списка.

Сайт НГУ не трогаем: вместо RatingListFetcher подставляется фейк.
"""

import logging
from typing import Any, cast

import pytest
from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import bot.notifications as notifications_module
import scheduler.main as scheduler_module
from db.postgres.dto import RatingEntry
from db.postgres.models import (
    Leaderboard,
    NotificationOutbox,
    NotificationStatus,
    User,
    UserRating,
)
from parser.rating import ListFetchResult

URL_A = "https://abiturient.nsu.ru/bachelor?faculty=1&direction=1&condition=1&type=30"
URL_B = "https://abiturient.nsu.ru/bachelor?faculty=1&direction=2&condition=1&type=30"
APPLICANT = "1234567"
TELEGRAM_ID = 555


def _as_bot(fake: Any) -> Bot:
    """Фейк повторяет только send_message — этого рассылке достаточно."""
    return cast(Bot, fake)


class FakeBot:
    """Записывает отправленное вместо похода в Telegram."""

    def __init__(self, fail: bool = False):
        self.sent: list[tuple[int, str]] = []
        self.fail = fail

    async def send_message(self, chat_id: int, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram недоступен")
        self.sent.append((chat_id, text))


class FakeFetcher:
    """Отдаёт заранее заданные результаты по URL вместо запросов к сайту."""

    def __init__(self, results: dict[str, ListFetchResult | Exception]):
        self.results = results
        self.closed = False

    def open(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def fetch(self, url: str) -> ListFetchResult:
        result = self.results[url]
        if isinstance(result, Exception):
            raise result
        return result


def _ok(place: int, page_hash: str, applicant: str = APPLICANT) -> ListFetchResult:
    return ListFetchResult(
        entries=[RatingEntry(applicant, place, "Подано", "Общий конкурс")],
        page_hash=page_hash,
        direction_name="Физика",
    )


@pytest.fixture
def maker(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest.fixture
def scheduler_env(monkeypatch, maker):
    """Планировщик и рассылка ходят в тестовую базу, без пауз и без сайта."""
    monkeypatch.setattr(scheduler_module, "AsyncSessionLocal", maker)
    monkeypatch.setattr(notifications_module, "AsyncSessionLocal", maker)
    monkeypatch.setattr(scheduler_module, "_use_mock", lambda: False)
    monkeypatch.setattr(scheduler_module, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(notifications_module, "SEND_DELAY_SECONDS", 0)
    monkeypatch.setattr(
        scheduler_module, "fetch_all_leaderboard_urls", lambda degree: [URL_A, URL_B]
    )

    def set_results(results: dict[str, ListFetchResult | Exception]) -> FakeFetcher:
        fetcher = FakeFetcher(results)
        monkeypatch.setattr(
            scheduler_module, "RatingListFetcher", lambda max_refreshes: fetcher
        )
        return fetcher

    return set_results


async def _add_user(maker, telegram_id: int = TELEGRAM_ID, applicant=APPLICANT):
    async with maker() as s:
        user = User(telegram_id=telegram_id, applicant_id=applicant)
        s.add(user)
        await s.commit()
        return user.user_id


async def _outbox(maker) -> list[NotificationOutbox]:
    async with maker() as s:
        rows = await s.execute(select(NotificationOutbox).order_by(NotificationOutbox.id))
        return list(rows.scalars())


async def _hash(maker, url: str) -> str | None:
    async with maker() as s:
        row = await s.execute(select(Leaderboard).where(Leaderboard.url == url))
        lb = row.scalar_one_or_none()
        return lb.content_hash if lb else None


# ─── #285: уведомления не теряются ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_crash_after_commit_does_not_lose_notification(
    scheduler_env, maker, monkeypatch
):
    """Процесс упал сразу после записи мест в базу — до рассылки.

    Раньше уведомления жили только в памяти: место в базе уже новое, хэш
    совпадает, и следующий цикл ничего не замечал. Теперь уведомление ждёт
    в outbox и уходит следующим циклом.
    """
    await _add_user(maker)
    scheduler_env({URL_A: _ok(5, "hash-a"), URL_B: _ok(5, "hash-b", "7654321")})

    original = scheduler_module._process_leaderboard

    async def crash_after_commit(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("OOM посреди обхода")

    monkeypatch.setattr(scheduler_module, "_process_leaderboard", crash_after_commit)
    bot = FakeBot()
    first = await scheduler_module.run_sync_job(_as_bot(bot))

    assert first.failed
    assert bot.sent == []
    assert [r.status for r in await _outbox(maker)] == [NotificationStatus.pending]

    # Следующий цикл: списки те же, хэш совпадает, диффа нет — но уведомление
    # из очереди уходит.
    monkeypatch.setattr(scheduler_module, "_process_leaderboard", original)
    await scheduler_module.run_sync_job(_as_bot(bot))

    assert len(bot.sent) == 1
    chat_id, text = bot.sent[0]
    assert chat_id == TELEGRAM_ID
    assert APPLICANT in text and "Место: 5" in text
    rows = await _outbox(maker)
    assert [r.status for r in rows] == [NotificationStatus.sent]
    assert rows[0].sent_at is not None


@pytest.mark.asyncio
async def test_exception_mid_crawl_still_delivers_processed_lists(scheduler_env, maker):
    """Второй список бросил исключение: уведомление по первому уже ушло."""
    await _add_user(maker)
    scheduler_env({URL_A: _ok(3, "hash-a"), URL_B: RuntimeError("сайт упал")})
    bot = FakeBot()

    result = await scheduler_module.run_sync_job(_as_bot(bot))

    assert result.failed
    assert len(bot.sent) == 1
    assert [r.status for r in await _outbox(maker)] == [NotificationStatus.sent]


@pytest.mark.asyncio
async def test_failed_send_stays_in_outbox_and_is_retried(scheduler_env, maker):
    """Telegram недоступен: уведомление не выбрасывается, а ждёт повтора."""
    await _add_user(maker)
    scheduler_env({URL_A: _ok(4, "hash-a"), URL_B: _ok(1, "hash-b", "7654321")})

    await scheduler_module.run_sync_job(_as_bot(FakeBot(fail=True)))
    rows = await _outbox(maker)
    assert [r.status for r in rows] == [NotificationStatus.pending]
    assert rows[0].attempts >= 1
    assert "telegram" in (rows[0].last_error or "")

    bot = FakeBot()
    await scheduler_module.run_sync_job(_as_bot(bot))
    assert len(bot.sent) == 1
    assert [r.status for r in await _outbox(maker)] == [NotificationStatus.sent]


@pytest.mark.asyncio
async def test_place_change_is_sent_once(scheduler_env, maker):
    """Повторные циклы не дублируют уже отправленное."""
    await _add_user(maker)
    scheduler_env({URL_A: _ok(10, "h1"), URL_B: _ok(1, "hb", "7654321")})
    bot = FakeBot()
    await scheduler_module.run_sync_job(_as_bot(bot))
    await scheduler_module.run_sync_job(_as_bot(bot))
    assert len(bot.sent) == 1

    scheduler_env({URL_A: _ok(8, "h2"), URL_B: _ok(1, "hb", "7654321")})
    await scheduler_module.run_sync_job(_as_bot(bot))
    assert len(bot.sent) == 2
    assert "10 ➡️ 8" in bot.sent[1][1]


# ─── #286: ошибка списка ≠ пустой список ────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_list_keeps_hash_and_is_logged_as_error(
    scheduler_env, maker, caplog
):
    user_id = await _add_user(maker)
    # Оба списка уже видели раньше.
    async with maker() as s:
        s.add_all(
            [
                Leaderboard(url=URL_A, direction="Физика", content_hash="old-a"),
                Leaderboard(url=URL_B, direction="Химия", content_hash="old-b"),
            ]
        )
        await s.commit()

    scheduler_env(
        {
            URL_A: _ok(2, "new-a"),
            URL_B: ListFetchResult.failure("запрос: 500 Server Error"),
        }
    )
    with caplog.at_level(logging.ERROR, logger=scheduler_module.__name__):
        result = await scheduler_module.run_sync_job(_as_bot(FakeBot()))

    assert result.failed_lists == 1 and result.total_lists == 2
    assert result.failed  # 50% > порога
    assert any(
        r.levelno == logging.ERROR and "1 из 2" in r.getMessage() for r in caplog.records
    )
    # Упавший список не тронут: следующий цикл увидит его заново.
    assert await _hash(maker, URL_B) == "old-b"
    assert await _hash(maker, URL_A) == "new-a"
    async with maker() as s:
        ratings = (
            await s.execute(select(UserRating).where(UserRating.user_id == user_id))
        ).scalars()
        assert [r.place for r in ratings] == [2]


@pytest.mark.asyncio
async def test_failures_below_threshold_are_warning_and_cycle_not_failed(
    scheduler_env, maker, monkeypatch, caplog
):
    monkeypatch.setattr(scheduler_module, "FAILED_LISTS_THRESHOLD", 0.6)
    scheduler_env({URL_A: _ok(2, "a"), URL_B: ListFetchResult.failure("таймаут")})

    with caplog.at_level(logging.WARNING, logger=scheduler_module.__name__):
        result = await scheduler_module.run_sync_job(_as_bot(FakeBot()))

    assert result.failed_lists == 1
    assert not result.failed
    assert any(
        r.levelno == logging.WARNING and "1 из 2" in r.getMessage()
        for r in caplog.records
    )
    assert await _hash(maker, URL_B) is None


@pytest.mark.asyncio
async def test_all_lists_failed_does_not_report_success(scheduler_env, maker):
    """Раньше это было «Изменений нет» и рост интервала до 30 минут."""
    fetcher = scheduler_env(
        {
            URL_A: ListFetchResult.failure("csrf"),
            URL_B: ListFetchResult.failure("csrf"),
        }
    )
    result = await scheduler_module.run_sync_job(_as_bot(FakeBot()))

    assert result.failed and not result.changed
    assert fetcher.closed
    # Интервал после такого цикла не растёт.
    assert (
        scheduler_module.compute_next_interval(
            600, result.changed, 300, 1800, failed=result.failed
        )
        == 600
    )


@pytest.mark.asyncio
async def test_genuinely_empty_list_is_not_an_error(scheduler_env, maker):
    scheduler_env({URL_A: ListFetchResult(page_hash="e"), URL_B: _ok(1, "b")})
    result = await scheduler_module.run_sync_job(_as_bot(FakeBot()))
    assert result.failed_lists == 0
    assert not result.failed
