import time
import pytest
from timemarker import TimeMarker


def test_timemarker_initial_state():
    tm = TimeMarker()
    assert tm.start_time is None
    assert tm.marks == []


def test_timemarker_start_sets_start_time():
    tm = TimeMarker()
    tm.start()
    assert tm.start_time is not None
    assert isinstance(tm.start_time, float)


def test_timemarker_start_twice_raises():
    tm = TimeMarker()
    tm.start()
    with pytest.raises(RuntimeError):
        tm.start()


def test_timemarker_mark_without_start_raises():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.mark("test")


def test_timemarker_mark_records_label_and_elapsed_time():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.mark("A")
    assert len(tm.marks) == 1
    label, elapsed = tm.marks[0]
    assert label == "A"
    assert elapsed > 0


def test_timemarker_multiple_marks():
    tm = TimeMarker()
    tm.start()
    tm.mark("first")
    time.sleep(0.01)
    tm.mark("second")
    assert len(tm.marks) == 2
    assert tm.marks[0][0] == "first"
    assert tm.marks[1][0] == "second"
    assert tm.marks[0][1] <= tm.marks[1][1]


def test_timemarker_stop_without_start_raises():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.stop()


def test_timemarker_stop_returns_total_elapsed_time():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    elapsed_total = tm.stop()
    assert isinstance(elapsed_total, float)
    assert elapsed_total >= 0
    # After stop, mark and stop should no longer be allowed
    with pytest.raises(RuntimeError):
        tm.mark("after stop")
    with pytest.raises(RuntimeError):
        tm.stop()


def test_timemarker_repr_contains_start_time_and_marks():
    tm = TimeMarker()
    tm.start()
    tm.mark("rep test")
    rep = repr(tm)
    assert "TimeMarker" in rep
    assert "rep test" in rep


def test_timemarker_reset_clears_state():
    tm = TimeMarker()
    tm.start()
    tm.mark("reset test")
    tm.stop()
    tm.reset()
    assert tm.start_time is None
    assert tm.marks == []


def test_timemarker_can_start_after_reset():
    tm = TimeMarker()
    tm.start()
    tm.mark("test")
    tm.stop()
    tm.reset()
    tm.start()
    tm.mark("new start")
    assert len(tm.marks) == 1
    assert tm.marks[0][0] == "new start"


def test_timemarker_elapsed_time_monotonicity():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    time.sleep(0.01)
    tm.mark("m2")
    times = [t for _, t in tm.marks]
    assert all(earlier <= later for earlier, later in zip(times, times[1:]))


def test_timemarker_mark_accepts_various_label_types():
    tm = TimeMarker()
    tm.start()
    labels = ["str", None, 123, 12.5, (1, 2)]
    for lbl in labels:
        tm.mark(lbl)
    recorded_labels = [label for label, _ in tm.marks]
    assert recorded_labels == labels


def test_timemarker_stop_returns_elapsed_close_to_last_mark():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    time.sleep(0.01)
    tm.mark("m2")
    total_elapsed = tm.stop()
    assert total_elapsed >= tm.marks[-1][1]


def test_timemarker_cannot_mark_after_stop():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.mark("m2")


def test_timemarker_repr_empty_before_start():
    tm = TimeMarker()
    rep = repr(tm)
    assert "TimeMarker" in rep
    assert "start_time=None" in rep or "start_time" in rep


def test_stop_twice_raises():
    tm = TimeMarker()
    tm.start()
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.stop()


def test_repr_includes_marks_and_elapsed_times():
    tm = TimeMarker()
    tm.start()
    tm.mark("a")
    tm.mark("b")
    rep = repr(tm)
    assert "a" in rep
    assert "b" in rep
    # The repr usually will include floats as well
    for label, elapsed in tm.marks:
        assert str(elapsed) in rep


@pytest.mark.parametrize("sleep_time", [0, 0.001, 0.005])
def test_mark_elapsed_increases_with_sleep(sleep_time):
    tm = TimeMarker()
    tm.start()
    time.sleep(sleep_time)
    tm.mark("test")
    elapsed = tm.marks[-1][1]
    assert elapsed >= sleep_time


@pytest.mark.parametrize("sleep_time", [0, 0.001])
def test_stop_elapsed_is_larger_than_last_mark(sleep_time):
    tm = TimeMarker()
    tm.start()
    time.sleep(sleep_time)
    tm.mark("m")
    time.sleep(sleep_time)
    elapsed_stop = tm.stop()
    last_mark_elapsed = tm.marks[-1][1]
    assert elapsed_stop >= last_mark_elapsed

