import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    i = Interval()
    assert i.begin == 0
    assert i.end == 2**63 - 1 or i.end > 0  # maxsize depends on platform


def test_interval_creation_with_values():
    i = Interval(1, 5)
    assert i.begin == 1
    assert i.end == 5


def test_interval_length():
    i = Interval(3, 10)
    assert len(i) == 7


def test_interval_length_zero():
    i = Interval(5, 5)
    assert len(i) == 0


def test_interval_less_than():
    i1 = Interval(1, 4)
    i2 = Interval(3, 6)
    assert (i1 < i2) is True
    assert (i2 < i1) is False


def test_interval_less_equal():
    i1 = Interval(1, 4)
    i2 = Interval(1, 6)
    assert (i1 <= i2) is True
    assert (i2 <= i1) is False


def test_interval_greater_than():
    i1 = Interval(5, 10)
    i2 = Interval(3, 7)
    assert (i1 > i2) is True
    assert (i2 > i1) is False


def test_interval_greater_equal():
    i1 = Interval(5, 10)
    i2 = Interval(5, 15)
    assert (i1 >= i2) is False
    assert (i2 >= i1) is True


def test_interval_contains_true():
    outer = Interval(1, 10)
    inner = Interval(2, 5)
    assert inner in outer


def test_interval_contains_false_begin():
    outer = Interval(1, 10)
    inner = Interval(0, 5)
    assert not (inner in outer)


def test_interval_contains_false_end():
    outer = Interval(1, 10)
    inner = Interval(5, 15)
    assert not (inner in outer)


def test_combine_intervals_order_independence():
    a = Interval(1, 5)
    b = Interval(4, 10)
    res1 = combine_intervals(a, b)
    res2 = combine_intervals(b, a)
    assert res1 == Interval(1, 10)
    assert res1 == res2


def test_combine_intervals_non_overlapping():
    a = Interval(1, 3)
    b = Interval(5, 8)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 8


def test_combine_three_intervals():
    a = Interval(1, 2)
    b = Interval(3, 5)
    c = Interval(4, 8)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 8


def test_get_minimum_bounding_interval_order():
    a = Interval(1, 5)
    b = Interval(3, 10)
    bounding1 = get_minimum_bounding_interval(a, b)
    bounding2 = get_minimum_bounding_interval(b, a)
    assert bounding1 == Interval(1, 10)
    assert bounding1 == bounding2


def test_interval_add_operator():
    a = Interval(1, 3)
    b = Interval(2, 7)
    result = a + b
    assert isinstance(result, Interval)
    assert result.begin == 1
    assert result.end == 7


def test_interval_overlaps_true():
    a = Interval(1, 5)
    b = Interval(4, 10)
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True


def test_interval_overlaps_false_adjacent():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_interval_overlaps_false_separated():
    a = Interval(1, 3)
    b = Interval(5, 8)
    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_interval_touches_true():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_touches_false():
    a = Interval(1, 4)
    b = Interval(5, 10)
    assert not a.touches(b)
    assert not b.touches(a)


def test_get_overlap_contained():
    a = Interval(1, 10)
    b = Interval(3, 6)
    result = a._get_overlap(b)
    assert result == b


def test_get_overlap_left_overlap():
    a = Interval(3, 8)
    b = Interval(1, 5)
    overlap = a._get_overlap(b)
    assert overlap.begin == 3
    assert overlap.end == 5


def test_get_overlap_right_overlap():
    a = Interval(3, 8)
    b = Interval(5, 10)
    overlap = a._get_overlap(b)
    assert overlap.begin == 5
    assert overlap.end == 8


def test_remove_contained_interval_returns_empty():
    a = Interval(1, 10)
    b = Interval(0, 20)
    result = a.remove(b)
    assert result == []


def test_remove_no_overlap_returns_self():
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]


def test_remove_partial_overlap_left():
    a = Interval(3, 10)
    b = Interval(1, 5)
    result = a.remove(b)
    expected = [Interval(5, 10)]
    assert result == expected


def test_remove_partial_overlap_right():
    a = Interval(1, 5)
    b = Interval(3, 10)
    result = a.remove(b)
    expected = [Interval(1, 3)]
    assert result == expected


def test_remove_partial_overlap_middle():
    a = Interval(1, 10)
    b = Interval(4, 6)
    result = a.remove(b)
    expected = [Interval(1, 4), Interval(6, 10)]
    assert result == expected


def test_remove_partial_overlap_zero_length_segment_filtered():
    a = Interval(1, 5)
    b = Interval(1, 3)
    result = a.remove(b)
    # segment (1,1) is zero length and should be filtered out, only (3,5) remains
    expected = [Interval(3, 5)]
    assert result == expected

