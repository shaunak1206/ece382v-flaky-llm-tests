import time
import pytest
from timemarker import TimeMarker

def test_timemarker_context_manager_basic_usage():
    with TimeMarker() as tm:
        time.sleep(0.01)
    assert isinstance(tm.elapsed(), float)
    assert tm.elapsed() > 0

def test_timemarker_elapsed_without_context_manager():
    tm = TimeMarker()
    time.sleep(0.01)
    elapsed = tm.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed > 0

def test_timemarker_reset_resets_elapsed():
    tm = TimeMarker()
    time.sleep(0.01)
    first = tm.elapsed()
    assert first > 0
    tm.reset()
    second = tm.elapsed()
    assert second < first

def test_timemarker_elapsed_returns_float_seconds():
    tm = TimeMarker()
    time.sleep(0.05)
    elapsed = tm.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.05

def test_timemarker_multiple_context_entries():
    tm = TimeMarker()
    for _ in range(3):
        with tm:
            time.sleep(0.01)
    elapsed = tm.elapsed()
    assert elapsed > 0

def test_timemarker_elapsed_after_reset_and_sleep():
    tm = TimeMarker()
    with tm:
        time.sleep(0.005)
    tm.reset()
    time.sleep(0.01)
    elapsed = tm.elapsed()
    assert elapsed >= 0.01

def test_timemarker_repr_contains_elapsed():
    tm = TimeMarker()
    with tm:
        time.sleep(0.01)
    r = repr(tm)
    assert "TimeMarker" in r
    assert str(round(tm.elapsed(), 3)) in r

def test_timemarker_multiple_resets_result_in_fresh_time():
    tm = TimeMarker()
    time.sleep(0.01)
    t1 = tm.elapsed()
    tm.reset()
    time.sleep(0.01)
    t2 = tm.elapsed()
    tm.reset()
    time.sleep(0.005)
    t3 = tm.elapsed()
    assert t1 > 0
    assert t2 > 0
    assert t3 > 0
    # each reset should give smaller elapsed right after reset
    assert t3 < t2 < t1

def test_timemarker_can_be_used_multiple_times_as_context_manager():
    tm = TimeMarker()
    with tm:
        time.sleep(0.005)
    first_elapsed = tm.elapsed()
    with tm:
        time.sleep(0.005)
    second_elapsed = tm.elapsed()
    assert first_elapsed > 0
    assert second_elapsed > 0

def test_timemarker_elapsed_increases_with_time():
    tm = TimeMarker()
    start = tm.elapsed()
    time.sleep(0.005)
    later = tm.elapsed()
    assert later >= start

def test_timemarker_elapsed_never_returns_negative():
    tm = TimeMarker()
    time.sleep(0.01)
    assert tm.elapsed() >= 0

def test_timemarker_elapsed_is_consistent_after_reset():
    tm = TimeMarker()
    time.sleep(0.01)
    first = tm.elapsed()
    tm.reset()
    time.sleep(0.01)
    second = tm.elapsed()
    assert second < first or abs(second - first) > 0  # should be different since reset

def test_timemarker_context_manager_returns_self():
    tm = TimeMarker()
    with tm as tm2:
        assert tm2 is tm

def test_timemarker_reset_allows_reuse():
    tm = TimeMarker()
    with tm:
        time.sleep(0.005)
    tm.reset()
    with tm:
        time.sleep(0.005)
    assert tm.elapsed() > 0

def test_timemarker_elapsed_has_precision():
    tm = TimeMarker()
    time.sleep(0.05)
    e = tm.elapsed()
    assert round(e, 2) > 0

def test_timemarker_supports_quick_usage():
    with TimeMarker() as tm:
        pass
    assert tm.elapsed() >= 0

def test_timemarker_elapsed_does_not_include_reset_overhead():
    tm = TimeMarker()
    start = time.perf_counter()
    tm.reset()
    end = time.perf_counter()
    elapsed = tm.elapsed()
    # elapsed should be less than the real time diff of reset overhead + small margin
    assert elapsed < (end-start) + 0.01

def test_timemarker_repr_returns_string():
    tm = TimeMarker()
    r = repr(tm)
    assert isinstance(r, str)
    assert "TimeMarker" in r

def test_timemarker_elapsed_with_no_sleep():
    tm = TimeMarker()
    elapsed = tm.elapsed()
    assert elapsed >= 0

def test_timemarker_elapsed_is_monotonic_with_sleep():
    tm = TimeMarker()
    e1 = tm.elapsed()
    time.sleep(0.001)
    e2 = tm.elapsed()
    assert e2 >= e1
    time.sleep(0.001)
    e3 = tm.elapsed()
    assert e3 >= e2

def test_timemarker_reset_multiple_times():
    tm = TimeMarker()
    tm.reset()
    time.sleep(0.004)
    t1 = tm.elapsed()
    tm.reset()
    t2 = tm.elapsed()
    assert t2 < t1

