import time
import pytest
from timemarker import TimeMarker

def test_timemarker_instantiates():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)

def test_timemarker_initial_mark_returns_float_and_is_nonnegative():
    tm = TimeMarker()
    mark = tm.mark()
    assert isinstance(mark, float)
    assert mark >= 0

def test_timemarker_mark_increases_with_time():
    tm = TimeMarker()
    m1 = tm.mark()
    time.sleep(0.01)
    m2 = tm.mark()
    assert m2 >= m1

def test_timemarker_elapsed_returns_float_and_is_nonnegative():
    tm = TimeMarker()
    tm.mark()
    time.sleep(0.005)
    elapsed = tm.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed >= 0

def test_elapsed_without_mark_raises_runtimeerror():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.elapsed()

def test_elapsed_after_mark_increases_over_time():
    tm = TimeMarker()
    tm.mark()
    elapsed1 = tm.elapsed()
    time.sleep(0.01)
    elapsed2 = tm.elapsed()
    assert elapsed2 >= elapsed1

def test_elapsed_with_custom_reference_mark():
    tm = TimeMarker()
    ref = tm.mark()
    time.sleep(0.005)
    elapsed = tm.elapsed(ref)
    assert isinstance(elapsed, float)
    assert elapsed >= 0

def test_elapsed_with_future_reference_mark_returns_negative_or_zero():
    tm = TimeMarker()
    ref = time.perf_counter() + 10
    elapsed = tm.elapsed(ref)
    assert elapsed <= 0

def test_elapsed_with_invalid_reference_mark_type():
    tm = TimeMarker()
    tm.mark()
    with pytest.raises(TypeError):
        tm.elapsed("not-a-float")

def test_elapsed_with_invalid_mark_invalid_type_error():
    tm = TimeMarker()
    with pytest.raises(TypeError):
        tm.elapsed([])  # list, invalid type

def test_multiple_marks_return_increasing_values():
    tm = TimeMarker()
    marks = [tm.mark() for _ in range(5)]
    for i in range(4):
        assert marks[i+1] >= marks[i]

def test_elapsed_between_two_marks():
    tm = TimeMarker()
    m1 = tm.mark()
    time.sleep(0.01)
    m2 = tm.mark()
    elapsed = m2 - m1
    # We simulate and confirm that elapsed reported between these marks matches
    elapsed_func = tm.elapsed(m1)
    # Since elapsed_func is current time - m1, and m2 > current time or equal,
    # elapsed_func should be >= elapsed
    assert elapsed_func >= elapsed

def test_elapsed_with_mark_larger_than_current_time_returns_negative():
    tm = TimeMarker()
    future_mark = time.perf_counter() + 5
    elapsed = tm.elapsed(future_mark)
    assert elapsed <= 0

def test_mark_and_elapsed_quick_succession():
    tm = TimeMarker()
    m = tm.mark()
    elapsed = tm.elapsed(m)
    # elapsed should be small, close to 0 or slightly more
    assert 0 <= elapsed < 0.1

def test_elapsed_default_mark_and_custom_mark_consistency():
    tm = TimeMarker()
    m1 = tm.mark()
    time.sleep(0.001)
    m2 = tm.mark()
    elapsed_def = tm.elapsed()
    elapsed_custom = tm.elapsed(m1)
    # elapsed def is now - last mark (which is m2)
    # elapsed_custom is now - m1 which should be >= elapsed_def
    assert elapsed_custom >= elapsed_def

def test_mark_called_multiple_times_updates_internal_state():
    tm = TimeMarker()
    first_mark = tm.mark()
    time.sleep(0.005)
    later_mark = tm.mark()
    # elapsed now should be based on later_mark
    elapsed = tm.elapsed()
    assert elapsed >= 0
    # elapsed between first and later marks is difference
    diff = later_mark - first_mark
    assert diff >= 0

def test_elapsed_after_long_sleep():
    tm = TimeMarker()
    tm.mark()
    time.sleep(0.05)
    elapsed = tm.elapsed()
    assert elapsed >= 0.05

def test_elapsed_with_mark_eq_last_mark():
    tm = TimeMarker()
    m = tm.mark()
    elapsed1 = tm.elapsed(m)
    elapsed2 = tm.elapsed()
    # Both elapsed values from same point, elapsed1 <= elapsed2 because last mark might be >= m
    assert elapsed1 <= elapsed2

def test_elapsed_with_none_as_arg_raises_typeerror():
    tm = TimeMarker()
    tm.mark()
    with pytest.raises(TypeError):
        tm.elapsed(None)

def test_internal_mark_time_monotonicity():
    # This test hypothesizes TimeMarker uses monotonic time (assert by ordering)
    tm = TimeMarker()
    times = [tm.mark() for _ in range(100)]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

def test_elapsed_after_repeated_mark_and_elapsed():
    tm = TimeMarker()
    for _ in range(10):
        tm.mark()
        time.sleep(0.001)
        assert tm.elapsed() >= 0

def test_elapsed_with_invalid_mark_out_of_range():
    tm = TimeMarker()
    tm.mark()
    with pytest.raises(TypeError):
        # using int (could convert implicitly to float but test refusal to accept)
        tm.elapsed(123456789)

