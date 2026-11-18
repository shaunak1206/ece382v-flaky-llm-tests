import time
import pytest
from timemarker import TimeMarker

def test_timemarker_init():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)
    assert tm.marks == []

def test_mark_records_label_and_time():
    tm = TimeMarker()
    tm.mark("start")
    assert len(tm.marks) == 1
    label, timestamp = tm.marks[0]
    assert label == "start"
    assert isinstance(timestamp, float)

def test_mark_appends_multiple_marks_in_order():
    tm = TimeMarker()
    tm.mark("first")
    time.sleep(0.01)
    tm.mark("second")
    assert len(tm.marks) == 2
    labels = [m[0] for m in tm.marks]
    assert labels == ["first", "second"]
    times = [m[1] for m in tm.marks]
    assert times[0] < times[1]

def test_mark_accepts_empty_label():
    tm = TimeMarker()
    tm.mark("")
    label, _ = tm.marks[0]
    assert label == ""

def test_duration_before_any_marks_returns_zero_or_none():
    tm = TimeMarker()
    # Without marks duration should be 0 or None depending on implementation
    duration = tm.duration()
    assert duration is None or duration == 0

def test_duration_with_one_mark_is_zero():
    tm = TimeMarker()
    tm.mark("only")
    dur = tm.duration()
    assert isinstance(dur, (int, float))
    assert dur == 0 or dur < 1e-3  # should be effectively zero

def test_duration_returns_difference_between_first_and_last_mark():
    tm = TimeMarker()
    tm.mark("start")
    time.sleep(0.02)
    tm.mark("end")
    dur = tm.duration()
    assert dur > 0.01
    assert dur < 1

def test_duration_with_multiple_marks():
    tm = TimeMarker()
    tm.mark("a")
    time.sleep(0.01)
    tm.mark("b")
    time.sleep(0.01)
    tm.mark("c")
    dur = tm.duration()
    assert dur >= 0.02

def test_durations_between_adjacent_marks():
    tm = TimeMarker()
    tm.mark("a")
    time.sleep(0.01)
    tm.mark("b")
    time.sleep(0.01)
    tm.mark("c")
    diffs = tm.differences()
    assert isinstance(diffs, list)
    assert len(diffs) == 2
    assert all(isinstance(d, float) for d in diffs)
    assert all(d > 0 for d in diffs)

def test_differences_empty_with_less_than_two_marks():
    tm = TimeMarker()
    assert tm.differences() == []
    tm.mark("only")
    assert tm.differences() == []

def test_clear_removes_all_marks():
    tm = TimeMarker()
    tm.mark("a")
    tm.mark("b")
    tm.clear()
    assert tm.marks == []

def test_repr_contains_labels_and_times():
    tm = TimeMarker()
    tm.mark("start")
    repr_str = repr(tm)
    assert "start" in repr_str
    assert "TimeMarker" in repr_str

def test_mark_labels_and_times_are_correct_type():
    tm = TimeMarker()
    tm.mark("mylabel")
    label, timestamp = tm.marks[0]
    assert isinstance(label, str)
    assert isinstance(timestamp, float)

def test_mark_with_non_str_label_raises_typeerror():
    tm = TimeMarker()
    with pytest.raises(TypeError):
        tm.mark(123)  # assuming implementation enforces string labels

def test_multiple_mark_calls_do_not_overwrite_marks():
    tm = TimeMarker()
    tm.mark("first")
    first_len = len(tm.marks)
    tm.mark("second")
    second_len = len(tm.marks)
    assert second_len == first_len + 1

def test_duration_is_sum_of_differences():
    tm = TimeMarker()
    tm.mark("a")
    time.sleep(0.01)
    tm.mark("b")
    time.sleep(0.01)
    tm.mark("c")
    diffs = tm.differences()
    dur = tm.duration()
    assert abs(dur - sum(diffs)) < 0.001

def test_clear_and_mark_behavior():
    tm = TimeMarker()
    tm.mark("start")
    tm.clear()
    tm.mark("after_clear")
    assert len(tm.marks) == 1
    assert tm.marks[0][0] == "after_clear"
