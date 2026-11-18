import time
import pytest
from timemarker import TimeMarker

def test_timemarker_initialization_default():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)
    assert hasattr(tm, "mark")
    assert hasattr(tm, "last")
    assert hasattr(tm, "checkpoint")
    assert tm.last is None
    assert tm.checkpoint is None

def test_timemarker_initialization_with_start():
    tm = TimeMarker(start=True)
    assert tm.last is not None
    assert tm.checkpoint is not None
    # On initialization with start=True, mark() was not called so mark_count = 0
    assert tm.marks_count == 0

def test_mark_records_timestamps_and_messages():
    tm = TimeMarker(start=True)
    time.sleep(0.01)
    tm.mark("first")
    time.sleep(0.01)
    tm.mark("second")

    assert len(tm.marks) == 2
    assert tm.marks[0].message == "first"
    assert tm.marks[1].message == "second"

def test_mark_returns_time_diff():
    tm = TimeMarker(start=True)
    tm.mark("initial")
    time.sleep(0.01)
    diff = tm.mark("next")
    assert isinstance(diff, float)
    assert diff > 0

def test_last_and_checkpoint_are_updated():
    tm = TimeMarker(start=True)
    first_mark = tm.mark("first")
    last_before = tm.last
    checkpoint_before = tm.checkpoint

    time.sleep(0.01)
    next_mark = tm.mark("second")
    assert tm.last > last_before
    assert tm.checkpoint > checkpoint_before

def test_reset_clears_and_starts():
    tm = TimeMarker()
    tm.mark("m1")
    tm.mark("m2")
    count_before = len(tm.marks)

    tm.reset(start=True)
    assert len(tm.marks) == 0
    assert tm.last is not None
    assert tm.checkpoint is not None

def test_reset_without_start_clears_only():
    tm = TimeMarker(start=True)
    tm.mark("m1")
    tm.mark("m2")
    tm.reset()  # no start

    assert len(tm.marks) == 0
    assert tm.last is None
    assert tm.checkpoint is None

def test_repr_contains_message_and_count():
    tm = TimeMarker(start=True)
    tm.mark("test mark")
    repr_str = repr(tm)
    assert "test mark" in repr_str or "test" in repr_str
    assert str(tm.marks_count) in repr_str

def test_repr_no_marks():
    tm = TimeMarker()
    r = repr(tm)
    assert "marks_count=0" in r or "marks_count" in r

def test_marks_count_increments():
    tm = TimeMarker()
    assert tm.marks_count == 0
    tm.mark("m1")
    assert tm.marks_count == 1
    tm.mark("m2")
    assert tm.marks_count == 2

def test_mark_without_message():
    tm = TimeMarker(start=True)
    result = tm.mark()
    assert isinstance(result, float)
    assert tm.marks[-1].message is None

def test_exception_if_mark_called_before_start():
    tm = TimeMarker()
    # Initially last is None so time_diff will fail calculation
    # According to code logic, mark may raise or handle None safely, test behavior:
    with pytest.raises(TypeError):
        # Because last is None, subtraction will raise TypeError
        tm.mark("test")

def test_checkpoints_increasing():
    tm = TimeMarker(start=True)
    tm.mark("m1")
    cp1 = tm.checkpoint
    time.sleep(0.01)
    tm.mark("m2")
    cp2 = tm.checkpoint
    assert cp2 > cp1

def test_marks_are_immutable_namedtuples():
    tm = TimeMarker(start=True)
    tm.mark("msg")
    mark = tm.marks[-1]
    # Test that mark has attributes time and message and they are read-only
    assert hasattr(mark, "time")
    assert hasattr(mark, "message")
    with pytest.raises(AttributeError):
        mark.time = 123

def test_marks_sorted_by_time():
    tm = TimeMarker(start=True)
    tm.mark("m1")
    time.sleep(0.01)
    tm.mark("m2")
    times = [m.time for m in tm.marks]
    assert times == sorted(times)

def test_reset_resets_marks_count():
    tm = TimeMarker(start=True)
    tm.mark("m1")
    assert tm.marks_count == 1
    tm.reset()
    assert tm.marks_count == 0

def test_mark_message_can_be_anything():
    tm = TimeMarker(start=True)
    tm.mark(123)
    tm.mark(None)
    tm.mark(["list", "of", "strings"])
    assert tm.marks[-3].message == 123
    assert tm.marks[-2].message is None
    assert tm.marks[-1].message == ["list", "of", "strings"]

def test_repr_contains_classname():
    tm = TimeMarker()
    r = repr(tm)
    assert "TimeMarker" in r

def test_time_precision():
    tm = TimeMarker(start=True)
    tm.mark("start")
    t1 = tm.last
    time.sleep(0.005)
    tm.mark("middle")
    t2 = tm.last
    assert t2 - t1 >= 0.005

def test_mark_with_long_message():
    tm = TimeMarker(start=True)
    long_msg = "x" * 1000
    tm.mark(long_msg)
    assert tm.marks[-1].message == long_msg
