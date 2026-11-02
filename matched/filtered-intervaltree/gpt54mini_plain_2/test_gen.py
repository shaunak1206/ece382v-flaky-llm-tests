import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_constructor_uses_full_range():
    i = Interval()
    assert i.begin == 0
    assert i.end > i.begin
    assert len(i) == i.end - i.begin


def test_interval_len_returns_difference_between_end_and_begin():
    i = Interval(3, 11)
    assert len(i) == 8


def test_interval_ordering_is_based_on_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a


def test_interval_contains_other_interval_when_fully_enclosed():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer not in inner


def test_interval_add_returns_bounding_interval():
    a = Interval(5, 8)
    b = Interval(1, 3)
    result = a + b
    assert result == Interval(1, 8)


def test_combine_intervals_returns_bounding_interval():
    a = Interval(4, 9)
    b = Interval(1, 6)
    assert combine_intervals(a, b) == Interval(1, 9)


def test_combine_intervals_is_commutative():
    a = Interval(4, 9)
    b = Interval(1, 6)
    assert combine_intervals(a, b) == combine_intervals(b, a)


def test_get_minimum_bounding_interval_with_non_overlapping_intervals():
    a = Interval(4, 9)
    b = Interval(1, 6)
    assert get_minimum_bounding_interval(a, b) == Interval(1, 9)


def test_get_minimum_bounding_interval_with_reversed_inputs():
    a = Interval(10, 20)
    b = Interval(1, 5)
    assert get_minimum_bounding_interval(a, b) == Interval(1, 20)


def test_combine_three_intervals_returns_bounding_interval():
    a = Interval(10, 20)
    b = Interval(3, 8)
    c = Interval(0, 1)
    assert combine_three_intervals(a, b, c) == Interval(0, 20)


def test_combine_three_intervals_handles_unsorted_inputs():
    a = Interval(7, 9)
    b = Interval(1, 2)
    c = Interval(4, 6)
    assert combine_three_intervals(a, b, c) == Interval(1, 9)


def test_touches_detects_adjacent_intervals():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert a.touches(b)
    assert b.touches(a)


def test_overlaps_detects_true_overlap_but_not_touching():
    a = Interval(1, 5)
    b = Interval(4, 10)
    c = Interval(5, 8)

    assert a.overlaps(b)
    assert not a.overlaps(c)
    assert not c.overlaps(a)


def test_remove_non_overlapping_interval_returns_original_interval():
    a = Interval(1, 5)
    b = Interval(10, 12)
    assert a.remove(b) == [a]


def test_remove_interval_fully_contained_in_other_returns_empty_list():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert outer.remove(inner) == []


def test_remove_interval_produces_remaining_left_and_right_segments():
    a = Interval(1, 10)
    b = Interval(3, 7)
    assert a.remove(b) == [Interval(1, 3), Interval(7, 10)]


def test_remove_overlapping_interval_at_left_edge_produces_right_segment_only():
    a = Interval(1, 10)
    b = Interval(0, 4)
    assert a.remove(b) == [Interval(4, 10)]


def test_remove_overlapping_interval_at_right_edge_produces_left_segment_only():
    a = Interval(1, 10)
    b = Interval(7, 12)
    assert a.remove(b) == [Interval(1, 7)]
