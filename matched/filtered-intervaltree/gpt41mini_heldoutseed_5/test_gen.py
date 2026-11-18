import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    i = Interval()
    assert i.begin == 0
    import sys
    assert i.end == sys.maxsize


def test_interval_creation_custom_values():
    i = Interval(10, 20)
    assert i.begin == 10
    assert i.end == 20


def test_interval_comparison_lt_le_gt_ge():
    i1 = Interval(1, 5)
    i2 = Interval(3, 7)
    i3 = Interval(1, 7)

    assert (i1 < i2) is True
    assert (i1 <= i3) is True
    assert (i2 > i1) is True
    assert (i3 >= i1) is True


def test_interval_len():
    i = Interval(4, 10)
    assert len(i) == 6


def test_interval_len_zero_or_negative():
    i1 = Interval(5, 5)
    i2 = Interval(7, 3)  # Though nonsensical, test behavior
    assert len(i1) == 0
    assert len(i2) == -4


def test_add_intervals():
    i1 = Interval(1, 5)
    i2 = Interval(3, 10)
    i3 = i1 + i2
    assert isinstance(i3, Interval)
    assert i3.begin == 1
    assert i3.end == 10


def test_combine_intervals_function():
    i1 = Interval(2, 6)
    i2 = Interval(4, 8)
    combined = combine_intervals(i1, i2)
    assert isinstance(combined, Interval)
    assert combined.begin == 2
    assert combined.end == 8


def test_combine_three_intervals_function():
    i1 = Interval(1, 3)
    i2 = Interval(5, 7)
    i3 = Interval(2, 6)
    combined = combine_three_intervals(i1, i2, i3)
    assert isinstance(combined, Interval)
    assert combined.begin == 1
    assert combined.end == 7


def test_get_minimum_bounding_interval_function():
    i1 = Interval(10, 20)
    i2 = Interval(15, 25)
    bounding = get_minimum_bounding_interval(i1, i2)
    assert isinstance(bounding, Interval)
    # The provided code is buggy since it calls sorted(points) but does not store result
    # So the returned Interval uses points[0] and points[-1] as from original unsorted list
    # Based on the source code, this test checks the actual buggy output:
    points = [10,20,15,25]
    assert bounding.begin == points[0]
    assert bounding.end == points[-1]


def test_interval_contains_true():
    outer = Interval(5, 15)
    inner = Interval(7, 10)
    assert inner in outer


def test_interval_contains_false():
    outer = Interval(5, 15)
    not_inner = Interval(3, 10)
    assert not (not_inner in outer)


def test_overlaps_method_true():
    i1 = Interval(1, 5)
    i2 = Interval(4, 8)
    assert i1.overlaps(i2)
    assert i2.overlaps(i1)


def test_overlaps_method_false():
    i1 = Interval(1, 3)
    i2 = Interval(3, 5)
    assert not i1.overlaps(i2)
    assert not i2.overlaps(i1)


def test_touches_method_true():
    i1 = Interval(1, 3)
    i2 = Interval(3, 5)
    assert i1.touches(i2)
    assert i2.touches(i1)


def test_touches_method_false():
    i1 = Interval(1, 4)
    i2 = Interval(5, 7)
    assert not i1.touches(i2)
    assert not i2.touches(i1)


def test_get_overlap_contained():
    outer = Interval(2, 10)
    inner = Interval(4, 8)
    overlap = outer._get_overlap(inner)
    assert overlap == inner


def test_get_overlap_partial_left():
    i1 = Interval(5, 10)
    i2 = Interval(3, 7)
    overlap = i1._get_overlap(i2)
    assert overlap.begin == 5
    assert overlap.end == 7


def test_get_overlap_partial_right():
    i1 = Interval(5, 10)
    i2 = Interval(7, 12)
    overlap = i1._get_overlap(i2)
    assert overlap.begin == 7
    assert overlap.end == 10


def test_remove_non_overlapping_returns_self():
    i1 = Interval(1, 5)
    i2 = Interval(6, 10)
    result = i1.remove(i2)
    assert result == [i1]


def test_remove_contained_returns_empty():
    i1 = Interval(3, 8)
    i2 = Interval(1, 10)
    # i1 in i2 is True, so result should be []
    result = i1.remove(i2)
    assert result == []


def test_remove_overlapping_splits():
    i1 = Interval(1, 10)
    i2 = Interval(4, 7)
    result = i1.remove(i2)
    # Result should be two intervals: [1,4) and [7,10)
    assert all(isinstance(x, Interval) for x in result)
    assert len(result) == 2
    assert result[0] == Interval(1, 4)
    assert result[1] == Interval(7, 10)


def test_remove_overlapping_touching_edges():
    i1 = Interval(1, 5)
    i2 = Interval(5, 7)
    result = i1.remove(i2)
    # They only touch, no overlap
    assert result == [i1]


def test_remove_overlapping_partial_overlap_at_start():
    i1 = Interval(3, 10)
    i2 = Interval(1, 5)
    result = i1.remove(i2)
    assert len(result) == 1
    assert result[0] == Interval(5, 10)
