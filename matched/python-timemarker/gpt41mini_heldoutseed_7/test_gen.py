import pytest
import time
from timemarker import TimeMarker

def test_timemarker_initialization():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)
    assert tm._marks == []
    assert tm._start is None
    assert tm._last is None

def test_start_sets_start_and_last():
    tm = TimeMarker()
    tm.start()
    assert tm._start is not None
    assert tm._last is not None
    # start and last should be equal on start
    assert tm._start == tm._last

def test_start_reset_clears_existing_marks():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    assert len(tm._marks) == 1
    tm.start()
    assert tm._marks == []
    assert tm._start == tm._last

def test_mark_raises_if_not_started():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.mark("test")

def test_mark_records_mark_with_duration_and_cumulative():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.mark("m1")
    time.sleep(0.01)
    tm.mark("m2")
    assert len(tm._marks) == 2
    first_mark = tm._marks[0]
    second_mark = tm._marks[1]
    # Each mark has a name, a delta and a cumtime
    assert first_mark[0] == "m1"
    assert isinstance(first_mark[1], float)
    assert isinstance(first_mark[2], float)
    assert second_mark[0] == "m2"
    assert second_mark[1] > 0
    assert second_mark[2] > first_mark[2]

def test_mark_updates_last_timestamp():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    first_last = tm._last
    time.sleep(0.005)
    tm.mark("m2")
    # _last should increase after second mark
    assert tm._last > first_last

def test_elapsed_without_start_returns_zero():
    tm = TimeMarker()
    # elapsed should be 0 if not started
    assert tm.elapsed() == 0

def test_elapsed_returns_correct_duration():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    elapsed = tm.elapsed()
    assert elapsed >= 0.01

def test_total_time_with_no_marks_returns_elapsed():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    total = tm.total_time()
    elapsed = tm.elapsed()
    # total_time should equal elapsed if no marks yet
    assert abs(total - elapsed) < 0.001

def test_total_time_with_marks_returns_last_cumtime():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    time.sleep(0.01)
    tm.mark("m2")
    total = tm.total_time()
    # total_time is the cumtime of the last mark
    assert abs(total - tm._marks[-1][2]) < 0.0001

def test_str_representation_includes_marks(monkeypatch):
    tm = TimeMarker()
    tm.start()
    tm.mark("start")
    time.sleep(0.01)
    tm.mark("end")
    out = str(tm)
    assert "start" in out
    assert "end" in out
    assert "TimeMarker" in out

def test_reset_clears_marks_and_times():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    tm.reset()
    assert tm._marks == []
    assert tm._start is None
    assert tm._last is None

def test_repr_contains_class_name_and_mark_count():
    tm = TimeMarker()
    tm.start()
    tm.mark("a")
    r = repr(tm)
    assert "TimeMarker" in r
    assert "marks=1" in r

def test_mark_with_different_names_are_unique_and_in_order():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    time.sleep(0.005)
    tm.mark("m2")
    time.sleep(0.005)
    tm.mark("m3")
    names = [m[0] for m in tm._marks]
    assert names == ["m1", "m2", "m3"]

def test_multiple_starts_resets_each_time():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    first_start = tm._start
    tm.start()
    second_start = tm._start
    assert second_start != first_start
    assert tm._marks == []

def test_elapsed_after_reset_returns_zero():
    tm = TimeMarker()
    tm.start()
    tm.mark("m1")
    tm.reset()
    assert tm.elapsed() == 0

def test_mark_raise_if_name_none_or_empty():
    tm = TimeMarker()
    tm.start()
    # Accept empty strings as mark names? Likely allowed, just test None
    with pytest.raises(TypeError):
        tm.mark(None)

def test_marking_many_marks_records_all():
    tm = TimeMarker()
    tm.start()
    for i in range(10):
        tm.mark(f"m{i}")
        time.sleep(0.001)
    assert len(tm._marks) == 10
    names = [m[0] for m in tm._marks]
    assert names == [f"m{i}" for i in range(10)]

def test_total_time_matches_last_cumtime_after_many_marks():
    tm = TimeMarker()
    tm.start()
    for i in range(5):
        tm.mark(f"m{i}")
        time.sleep(0.001)
    total = tm.total_time()
    assert abs(total - tm._marks[-1][2]) < 0.001
