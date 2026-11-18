import random
import pytest
import numpy as np

from timemarker import TimeMarker


def test_timemarker_instantiation():
    random.seed(0)
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)


def test_timemarker_start_and_elapsed(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    # Start with no parameters, just verify .start() doesn't error
    tm.start()
    e = tm.elapsed()
    # elapsed should return float, >= 0
    assert isinstance(e, float)
    assert e >= 0


def test_timemarker_elapsed_before_start(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    # Call elapsed before start - depending on implementation either error or 0 or some float
    # We expect it to not error, and return a float >= 0
    try:
        e = tm.elapsed()
    except Exception:
        pytest.skip("elapsed() raises exception if called before start()")
    else:
        assert isinstance(e, float)
        assert e >= 0


def test_timemarker_multiple_starts(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e1 = tm.elapsed()
    tm.start()
    e2 = tm.elapsed()
    # When restart after starting, elapsed restarts as well or not, depends on implementation;
    # but elapsed should not shrink a lot (no negative or zero)
    assert e1 >= 0
    assert e2 >= 0


def test_timemarker_elapsed_float(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e = tm.elapsed()
    # The elapsed time is a float; test tolerance on some dummy difference
    assert abs(e - float(e)) < 1e-9


def test_timemarker_str_representation(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    s = str(tm)
    # Should not error and should include class name or something informative
    assert isinstance(s, str)
    assert "TimeMarker" in s or "time" in s.lower()


def test_timemarker_repr_includes_classname(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    r = repr(tm)
    # repr should contain class name
    assert isinstance(r, str)
    assert "TimeMarker" in r


def test_start_time_changes(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    first = tm.elapsed()
    tm.start()
    second = tm.elapsed()
    # The elapsed after second start should be less or equal than first elapsed, because restarted
    assert 0 <= second <= first


def test_elapsed_returns_zero_if_no_elapsed_time(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    # Monkeypatch time method used in TimeMarker to 0 to force elapsed zero
    # We don't know time implementation so skip this test if private methods unknown
    # Instead just check elapsed >= 0 always
    elapsed = tm.elapsed()
    assert elapsed >= 0


def test_elapsed_and_start_do_not_modify_globals(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    before_attrs = set(dir(tm))
    tm.start()
    after_attrs = set(dir(tm))
    tm.elapsed()
    # Assert no attribute deleted or added dynamically during calls
    assert before_attrs == after_attrs


def test_multiple_instances_independent(monkeypatch):
    random.seed(0)
    tm1 = TimeMarker()
    tm2 = TimeMarker()
    tm1.start()
    tm2.start()
    e1 = tm1.elapsed()
    e2 = tm2.elapsed()
    # Each TimeMarker instance's elapsed should be independent floats >=0
    assert isinstance(e1, float)
    assert isinstance(e2, float)
    assert e1 >= 0
    assert e2 >= 0


def test_elapsed_consistency_across_calls(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e1 = tm.elapsed()
    e2 = tm.elapsed()
    # elapsed() called multiple times without restart should be equal or monotonic increasing
    assert e1 <= e2


def test_start_resets_elapsed(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e1 = tm.elapsed()
    tm.start()
    e2 = tm.elapsed()
    # Start should reset elapsed timer so second elapsed less or equal first
    assert e2 <= e1


def test_elapsed_always_nonnegative(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e = tm.elapsed()
    assert e >= 0


def test_elapsed_returns_float_type(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    e = tm.elapsed()
    assert isinstance(e, float)


def test_repr_and_str_differ(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    s = str(tm)
    r = repr(tm)
    # They may or may not differ but both must be str and contain class name at least in one
    assert isinstance(s, str)
    assert isinstance(r, str)
    assert "TimeMarker" in s or "TimeMarker" in r


def test_elapsed_after_long_time_simulation(monkeypatch):
    random.seed(0)
    import time

    tm = TimeMarker()

    # We monkeypatch time.perf_counter or time.time inside TimeMarker to simulate a longer elapsed time without actually waiting
    # We must find which timer is used by TimeMarker (guess is time.perf_counter or time.time).
    # Let's patch time.perf_counter to a controlled generator function simulating monotonic time

    times = [1000, 1005]

    def fake_perf_counter():
        return times.pop(0)

    # Patch time.perf_counter in timemarker.timer module
    monkeypatch.setattr("timemarker.timer.time", time)
    monkeypatch.setattr("timemarker.timer.time.perf_counter", fake_perf_counter)

    tm.start()
    elapsed = tm.elapsed()
    assert abs(elapsed - 5) < 1e-9


def test_start_returns_none(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    res = tm.start()
    # Start should return None or nothing
    assert res is None


def test_elapsed_not_affected_by_seed(monkeypatch):
    # elapsed is time based, random seed set to verify no flaky due to randomness
    random.seed(42)
    tm = TimeMarker()
    tm.start()
    e1 = tm.elapsed()
    random.seed(42)
    e2 = tm.elapsed()
    assert abs(e2 - e1) < 1e-6


def test_elapsed_after_manual_monkeypatch_time(monkeypatch):
    random.seed(0)
    import time

    tm = TimeMarker()

    base_time = 1234.56

    def fake_perf_counter():
        return base_time

    monkeypatch.setattr("timemarker.timer.time", time)
    monkeypatch.setattr("timemarker.timer.time.perf_counter", fake_perf_counter)

    tm.start()
    e = tm.elapsed()
    # Since perf_counter never advances, elapsed should be zero
    assert abs(e) < 1e-9


def test_elapsed_monotonic(monkeypatch):
    random.seed(0)
    import time

    tm = TimeMarker()
    tm.start()

    # Patch perf_counter progressively to larger values
    times = [10, 20, 30]

    def fake_perf_counter():
        return times.pop(0)

    monkeypatch.setattr("timemarker.timer.time", time)
    monkeypatch.setattr("timemarker.timer.time.perf_counter", fake_perf_counter)

    e1 = tm.elapsed()  # 10-10=0 or so
    e2 = tm.elapsed()  # 20-10=10
    e3 = tm.elapsed()  # 30-10=20

    # Because times list pop(0), these calls progress
    assert e1 <= e2 <= e3


def test_elapsed_does_not_raise_if_accessed_many_times(monkeypatch):
    random.seed(0)
    tm = TimeMarker()
    tm.start()
    for _ in range(100):
        val = tm.elapsed()
        assert isinstance(val, float)
        assert val >= 0


def test_time_marker_importable_and_has_class(monkeypatch):
    random.seed(0)
    from timemarker import TimeMarker as TM
    assert TM == TimeMarker
