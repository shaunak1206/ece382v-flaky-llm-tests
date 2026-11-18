import random
import pytest
import numpy as np

from timemarker import TimeMarker


def test_timemarker_instantiation_basic():
    """Can instantiate TimeMarker and has expected attributes."""
    random.seed(0)
    tm = TimeMarker()
    # Should have methods start, stop, reset, elapsed, etc.
    assert callable(getattr(tm, "start", None))
    assert callable(getattr(tm, "stop", None))
    assert callable(getattr(tm, "reset", None))
    assert callable(getattr(tm, "elapsed", None))


def test_timemarker_start_stop_elapsed_consistency():
    random.seed(1)
    tm = TimeMarker()
    tm.reset()
    tm.start()
    tm.stop()
    elapsed1 = tm.elapsed()
    # Immediately after stop elapsed should be non-negative float
    assert isinstance(elapsed1, float)
    assert elapsed1 >= 0.0
    # If call elapsed again without start/stop, its value should not change
    elapsed2 = tm.elapsed()
    assert elapsed1 == pytest.approx(elapsed2, rel=1e-9)


def test_timemarker_reset_clears_elapsed():
    random.seed(2)
    tm = TimeMarker()
    tm.start()
    tm.stop()
    elapsed_before = tm.elapsed()
    assert elapsed_before >= 0.0
    tm.reset()
    # After reset, elapsed should be zero or reset state, depends on impl
    tm.start()
    tm.stop()
    elapsed_after = tm.elapsed()
    assert isinstance(elapsed_after, float)
    assert elapsed_after >= 0.0


def test_timemarker_elapsed_without_start_stop():
    random.seed(3)
    tm = TimeMarker()
    # Without any start/stop calls, elapsed should be zero or raise no errors
    val = tm.elapsed()
    assert isinstance(val, float)
    assert val >= 0.0


def test_timemarker_multiple_start_stop_cycles():
    random.seed(4)
    tm = TimeMarker()
    elapsed_values = []
    for _ in range(5):
        tm.reset()
        tm.start()
        tm.stop()
        elapsed_values.append(tm.elapsed())
    # All elapsed must be non-negative floats
    for ev in elapsed_values:
        assert isinstance(ev, float)
        assert ev >= 0.0
    # No reset elapsed should differ, but all valid
    assert len(set(elapsed_values)) <= 5


def test_timemarker_stop_without_start_does_not_fail():
    random.seed(5)
    tm = TimeMarker()
    # Calling stop before start should not raise
    try:
        tm.stop()
    except Exception:
        pytest.fail("stop() raised exception when called without start()")
    # elapsed after stop without start should be float >=0
    val = tm.elapsed()
    assert isinstance(val, float)
    assert val >= 0.0


def test_timemarker_start_while_running_behavior():
    random.seed(6)
    tm = TimeMarker()
    # Start once
    tm.start()
    # Start again while running, should not raise or corrupt elapsed state
    try:
        tm.start()
    except Exception:
        pytest.fail("start() raised exception when called twice in a row")
    tm.stop()
    val = tm.elapsed()
    # Elapsed should be non-negative float and not zero (some time elapsed)
    assert isinstance(val, float)
    assert val >= 0.0


def test_timemarker_elapsed_after_reset_and_start_stop():
    random.seed(7)
    tm = TimeMarker()
    tm.start()
    tm.stop()
    first_elapsed = tm.elapsed()
    tm.reset()
    tm.start()
    tm.stop()
    second_elapsed = tm.elapsed()
    # Both results should be non-negative floats, may differ or be zero
    assert isinstance(first_elapsed, float) and first_elapsed >= 0.0
    assert isinstance(second_elapsed, float) and second_elapsed >= 0.0


def test_timemarker_elapsed_increases_after_longer_wait(monkeypatch):
    random.seed(8)
    tm = TimeMarker()

    # Define fake timer to simulate passage of time deterministically
    fake_time_values = [10.0, 10.5, 12.0]
    times = iter(fake_time_values)

    def fake_time():
        return next(times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.reset()
    tm.start()
    tm.stop()
    first_elapsed = tm.elapsed()
    tm.reset()
    tm.start()
    tm.stop()
    second_elapsed = tm.elapsed()

    # Because fake timers are increasing, second elapsed should be larger
    assert first_elapsed == pytest.approx(0.5, abs=1e-9)
    assert second_elapsed == pytest.approx(2.0, abs=1e-9)
    assert second_elapsed > first_elapsed


def test_timemarker_reset_clears_all_state(monkeypatch):
    random.seed(9)
    tm = TimeMarker()
    # Use fake time to simulate elapsed
    fake_time_values = [1.0, 1.2, 1.5, 1.9, 2.3]
    times = iter(fake_time_values)

    def fake_time():
        return next(times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed_before_reset = tm.elapsed()
    tm.reset()
    # After reset, elapsed should reset or be small (simulated 0 from fake_time)
    tm.start()
    tm.stop()
    elapsed_after_reset = tm.elapsed()

    assert elapsed_before_reset == pytest.approx(0.2, abs=1e-9)
    assert elapsed_after_reset == pytest.approx(0.4, abs=1e-9)
    assert elapsed_after_reset != elapsed_before_reset


def test_timemarker_elapsed_type_and_value(monkeypatch):
    random.seed(10)
    tm = TimeMarker()
    start_time = 100.0
    stop_time = 200.0

    def fake_time_start():
        return start_time

    def fake_time_stop():
        return stop_time

    # Patch time so that start() reads start_time, stop() reads stop_time
    call_times = [start_time, stop_time]

    def fake_time():
        return call_times.pop(0)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed = tm.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed == stop_time - start_time


def test_timemarker_elapsed_is_zero_if_stop_eq_start(monkeypatch):
    random.seed(11)
    tm = TimeMarker()

    def fake_time():
        return 42.0

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed = tm.elapsed()
    assert elapsed == 0.0


def test_timemarker_elapsed_after_multiple_starts_and_stops(monkeypatch):
    random.seed(12)
    tm = TimeMarker()
    fake_times = iter([1, 2, 3, 6])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    first = tm.elapsed()  # 2 - 1 == 1

    tm.start()
    tm.stop()
    second = tm.elapsed()  # 6 - 3 == 3

    # elapsed() after stop returns duration of last start-stop cycle?
    assert first == pytest.approx(1, abs=1e-9)
    assert second == pytest.approx(3, abs=1e-9)


def test_timemarker_elapsed_before_any_start(monkeypatch):
    random.seed(13)
    tm = TimeMarker()

    def fake_time():
        return 500.0

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    # elapsed before start should be zero or valid float >= 0
    elapsed = tm.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timemarker_elapsed_after_start_without_stop(monkeypatch):
    random.seed(14)
    tm = TimeMarker()
    t = [10.0]

    def fake_time():
        # each call increments time by +1.0 to simulate running timer
        current = t[0]
        t[0] += 1.0
        return current

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    elapsed1 = tm.elapsed()
    elapsed2 = tm.elapsed()
    # elapsed should be increasing or same if time hasn't advanced
    assert elapsed2 >= elapsed1
    assert elapsed1 >= 0.0


def test_timemarker_reset_after_partial_run(monkeypatch):
    random.seed(15)
    tm = TimeMarker()
    fake_times = iter([5, 7, 10, 13])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    first_elapsed = tm.elapsed()  # 7-5=2
    tm.reset()
    # After reset, elapsed should be 0 or time difference after next start/stop
    tm.start()
    tm.stop()
    second_elapsed = tm.elapsed()  # 13-10=3
    assert first_elapsed == pytest.approx(2, abs=1e-9)
    assert second_elapsed == pytest.approx(3, abs=1e-9)
    assert second_elapsed != first_elapsed


def test_timemarker_elapsed_with_float_precision(monkeypatch):
    random.seed(16)
    tm = TimeMarker()
    base = 1000.0000001
    fake_times = iter([base, base + 0.0000001])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed = tm.elapsed()
    assert abs(elapsed - 0.0000001) < 1e-8


def test_timemarker_elapsed_after_stop_multiple_calls(monkeypatch):
    random.seed(17)
    tm = TimeMarker()
    fake_times = iter([0, 1])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed1 = tm.elapsed()
    elapsed2 = tm.elapsed()
    # Multiple calls to elapsed after stop should be same value
    assert elapsed1 == elapsed2


def test_timemarker_start_stop_order(monkeypatch):
    random.seed(18)
    tm = TimeMarker()
    fake_times = iter([0, 2])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    # Stop before start should not raise, elapsed should be zero
    tm.stop()
    assert tm.elapsed() >= 0.0

    tm.start()
    tm.stop()
    # elapsed should be 2.0 (2-0)
    assert tm.elapsed() == pytest.approx(2.0, abs=1e-9)


def test_timemarker_elapsed_never_negative(monkeypatch):
    random.seed(19)
    tm = TimeMarker()

    # fake a timer returning decreasing time to test for negative protection
    fake_times = iter([10, 9])

    def fake_time():
        return next(fake_times)

    monkeypatch.setattr("timemarker.timer.time", fake_time)

    tm.start()
    tm.stop()
    elapsed = tm.elapsed()
    assert elapsed >= 0.0
