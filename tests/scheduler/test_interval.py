"""Тесты адаптивного интервала опроса конкурсных списков."""

from scheduler.main import compute_next_interval

MIN_S = 300
MAX_S = 1800


def test_no_change_backs_off_and_caps_at_max():
    # Старт с максимума: без изменений остаёмся на максимуме (×2 упирается в потолок).
    assert compute_next_interval(1800, False, MIN_S, MAX_S) == 1800
    # С меньшего значения без изменений — реже (×2), но не выше максимума.
    assert compute_next_interval(600, False, MIN_S, MAX_S) == 1200
    assert compute_next_interval(1200, False, MIN_S, MAX_S) == 1800


def test_change_speeds_up():
    assert compute_next_interval(1800, True, MIN_S, MAX_S) == 900
    assert compute_next_interval(900, True, MIN_S, MAX_S) == 450


def test_change_floored_at_min():
    # 450 ÷ 2 = 225 → упирается в минимум 300.
    assert compute_next_interval(450, True, MIN_S, MAX_S) == 300
    assert compute_next_interval(300, True, MIN_S, MAX_S) == 300


def test_no_change_from_min_goes_up():
    assert compute_next_interval(300, False, MIN_S, MAX_S) == 600
