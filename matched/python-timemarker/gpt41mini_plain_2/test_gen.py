import time
import pytest
from timemarker import TimeMarker

def test_initialization_without_tags():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)
    # Initially no marks
    assert list(tm.marks()) == []

def test_initialization_with_tags():
    tags = ["start", "middle", "end"]
    tm = TimeMarker(tags)
    assert tm._tags == tags
    # Since no marks yet, .marks() is empty
    assert list(tm.marks()) == []

def test_mark_recording_single_tag():
    tm = TimeMarker()
    tm.mark("tag1")
    marks = list(tm.marks())
    assert len(marks) == 1
    assert marks[0][0] == "tag1"
    assert isinstance(marks[0][1], float)

def test_mark_recording_multiple_tags():
    tags = ["a", "b"]
    tm = TimeMarker(tags)
    tm.mark("a")
    tm.mark("b")
    tm.mark("a")
    records = list(tm.marks())
    assert len(records) == 3
    # Tags appear in order marked
    assert records[0][0] == "a"
    assert records[1][0] == "b"
    assert records[2][0] == "a"

def test_mark_with_invalid_tag_raises():
    tm = TimeMarker(["tag1"])
    with pytest.raises(ValueError):
        tm.mark("invalid")

def test_tags_property():
    tags = ["foo", "bar"]
    tm = TimeMarker(tags)
    assert list(tm.tags()) == tags

def test_marks_property_content_and_type():
    tm = TimeMarker()
    tm.mark("test")
    for tag, timestamp in tm.marks():
        assert isinstance(tag, str)
        assert isinstance(timestamp, float)

def test_times_and_tags_methods_consistency():
    tm = TimeMarker(["x", "y"])
    tm.mark("x")
    tm.mark("y")
    tags = list(tm.tags())
    assert tags == ["x", "y"]
    marks = list(tm.marks())
    assert len(marks) == 2
    mark_tags = [t[0] for t in marks]
    assert set(mark_tags) <= set(tags)

def test_mark_and_time_order():
    tm = TimeMarker()
    tm.mark("tag1")
    time.sleep(0.01)
    tm.mark("tag2")
    marks = list(tm.marks())
    assert marks[0][1] < marks[1][1]

def test_repr_contains_tags_and_counts():
    tags = ["a", "b", "c"]
    tm = TimeMarker(tags)
    tm.mark("a")
    tm.mark("b")
    r = repr(tm)
    assert all(t in r for t in tags)
    assert "2 marks" in r or "marks=2" in r or "2" in r  # minimal sanity

def test_clear_removes_all_marks():
    tm = TimeMarker()
    tm.mark("t1")
    tm.mark("t2")
    tm.clear()
    assert list(tm.marks()) == []

def test_marking_with_no_tags_allowed_any_tag():
    tm = TimeMarker()
    tm.mark("any_tag")
    marks = list(tm.marks())
    assert marks[0][0] == "any_tag"

def test_clear_before_any_mark_does_not_error():
    tm = TimeMarker()
    tm.clear()
    assert list(tm.marks()) == []

def test_elapsed_time_between_two_marks():
    tm = TimeMarker()
    tm.mark("start")
    time.sleep(0.01)
    tm.mark("end")
    marks = list(tm.marks())
    elapsed = marks[1][1] - marks[0][1]
    assert elapsed > 0

def test_iterability_of_marks():
    tm = TimeMarker()
    tm.mark("a")
    tm.mark("b")
    it = iter(tm.marks())
    first = next(it)
    second = next(it)
    with pytest.raises(StopIteration):
        next(it)
    assert first[0] == "a"
    assert second[0] == "b"

def test_mark_with_none_tag_raises():
    tm = TimeMarker(["tag1"])
    with pytest.raises(ValueError):
        tm.mark(None)

def test_marking_many_tags_quickly():
    tm = TimeMarker(["t1", "t2", "t3"])
    for i in range(100):
        tm.mark("t1")
        tm.mark("t2")
        tm.mark("t3")
    marks = list(tm.marks())
    assert len(marks) == 300

def test_tags_non_list_raises_or_accepted():
    # The source does not clarify if tags param can be other than list.
    # Try passing tuple:
    tags = ("a", "b")
    tm = TimeMarker(tags)
    assert list(tm.tags()) == list(tags)
    # Try passing None:
    with pytest.raises(TypeError):
        TimeMarker(None)

def test_repr_empty_marks():
    tm = TimeMarker()
    repr_str = repr(tm)
    assert "marks" in repr_str

def test_repr_after_clear_has_zero_marks():
    tm = TimeMarker(["tag"])
    tm.mark("tag")
    tm.clear()
    r = repr(tm)
    assert "0" in r

def test_marking_tag_not_in_tags_if_tags_provided_raises():
    tm = TimeMarker(["t1", "t2"])
    with pytest.raises(ValueError):
        tm.mark("t3")

