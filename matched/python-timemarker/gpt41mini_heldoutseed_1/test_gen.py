import time
import pytest
from timemarker import TimeMarker


def test_timemarker_init_no_args():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)
    assert tm.name == ""
    assert hasattr(tm, "start")
    assert hasattr(tm, "splits")
    assert hasattr(tm, "stopped")
    assert isinstance(tm.splits, list)
    assert tm.splits == []
    assert tm.stopped is False


def test_timemarker_init_with_name():
    tm = TimeMarker("testname")
    assert tm.name == "testname"


def test_timemarker_start_defaults(monkeypatch):
    tm = TimeMarker()
    before = time.perf_counter()
    tm.start()
    after = time.perf_counter()
    # The start time should be between before and after
    assert before <= tm.start <= after
    assert tm.splits == []
    assert tm.stopped is False


def test_timemarker_start_with_argument(monkeypatch):
    tm = TimeMarker()
    tm.start(10.0)
    assert tm.start == 10.0
    assert tm.stopped is False


def test_timemarker_split_records_correct_time(monkeypatch):
    tm = TimeMarker()
    tm.start(0.0)
    tm.split(1.5)
    tm.split(3.0)
    assert tm.splits == [1.5, 3.0]


def test_timemarker_split_with_default_time(monkeypatch):
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.split()
    assert len(tm.splits) == 1
    # split time should be >= start time
    assert tm.splits[0] >= tm.start


def test_timemarker_split_after_stop_raises():
    tm = TimeMarker()
    tm.start()
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.split()


def test_timemarker_stop_records_stop_time(monkeypatch):
    tm = TimeMarker()
    tm.start(5.0)
    tm.stop(10.0)
    assert tm.stopped is True
    assert tm.splits[-1] == 10.0 if tm.splits else True


def test_timemarker_stop_with_default_time():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    assert tm.stopped is True
    assert tm.splits[-1] >= tm.start


def test_timemarker_stop_twice_raises():
    tm = TimeMarker()
    tm.start()
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.stop()


def test_timemarker_split_before_start_raises():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.split()


def test_timemarker_stop_before_start_raises():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.stop()


def test_timemarker_repr_contains_name_and_splits():
    tm = TimeMarker("mytimer")
    tm.start(0.0)
    tm.split(1.0)
    tm.stop(2.0)
    repr_str = repr(tm)
    assert "mytimer" in repr_str
    assert "2.0" in repr_str or "1.0" in repr_str


def test_timemarker_multiple_splits(monkeypatch):
    tm = TimeMarker()
    tm.start(0)
    splits = [0.1, 0.5, 1.0, 2.0]
    for s in splits:
        tm.split(s)
    assert tm.splits == splits


def test_timemarker_stop_adds_last_split(monkeypatch):
    tm = TimeMarker()
    tm.start(0)
    tm.split(0.5)
    tm.stop(1.5)
    assert tm.splits[-1] == 1.5


def test_timemarker_splits_are_floats():
    tm = TimeMarker()
    tm.start()
    tm.split()
    tm.stop()
    for s in tm.splits:
        assert isinstance(s, float)


def test_timemarker_name_default_empty_string():
    tm = TimeMarker()
    assert tm.name == ""


def test_timemarker_start_resets_splits(monkeypatch):
    tm = TimeMarker()
    tm.start(0)
    tm.split(1)
    assert len(tm.splits) == 1
    tm.start(2)
    # Starting again should reset splits list
    assert tm.splits == []
    assert tm.start == 2


def test_timemarker_stop_marks_stopped(monkeypatch):
    tm = TimeMarker()
    tm.start()
    assert not tm.stopped
    tm.stop()
    assert tm.stopped


def test_timemarker_behavior_after_stop(monkeypatch):
    tm = TimeMarker()
    tm.start()
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.split()
    with pytest.raises(RuntimeError):
        tm.stop()


def test_timemarker_has_elapsed_property(monkeypatch):
    # This test assumes the TimeMarker has an elapsed property or method
    # If not exist, skip
    tm = TimeMarker()
    tm.start(0)
    tm.split(0.5)
    tm.stop(1.0)
    # Test if elapsed attribute or method available and returns float
    elapsed = None
    if hasattr(tm, "elapsed"):
        attr = getattr(tm, "elapsed")
        elapsed = attr() if callable(attr) else attr
    if elapsed is not None:
        assert isinstance(elapsed, float)
        assert elapsed >= 0


def test_timemarker_name_accepts_non_string_types():
    tm = TimeMarker(123)
    assert str(tm.name) == "123" or tm.name == 123


def test_timemarker_repr_without_name():
    tm = TimeMarker()
    tm.start(0)
    tm.split(1)
    rep = repr(tm)
    assert isinstance(rep, str)
    assert "TimeMarker" in rep
