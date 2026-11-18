import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_constructor_uses_zero_and_maxsize():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == sys.maxsize


def test_interval_len_returns_difference_between_end_and_begin():
    interval = Interval(3, 10)
    assert len(interval) == 7


def test_interval_ordering_compares_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a


def test_interval_contains_true_when_inner_interval_is_fully_contained():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_interval_extends_beyond_bounds():
    outer = Interval(1, 10)
    overlapping = Interval(3, 12)
    assert overlapping not in outer


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 8)
    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_interval_touches_true_when_end_equals_other_begin():
    a = Interval(1, 5)
    b = Interval(5, 8)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_add_returns_minimum_bounding_interval():
    a = Interval(5, 10)
    b = Interval(1, 8)
    result = a + b
    assert result == Interval(1, 10)


def test_combine_intervals_returns_minimum_bounding_interval():
    a = Interval(5, 10)
    b = Interval(1, 8)
    result = combine_intervals(a, b)
    assert result == Interval(1, 10)


def test_combine_intervals_is_commutative():
    a = Interval(5, 10)
    b = Interval(1, 8)
    assert combine_intervals(a, b) == combine_intervals(b, a)


def test_combine_three_intervals_uses_first_and_last_arguments_without_sorting():
    a = Interval(10, 20)
    b = Interval(1, 2)
    c = Interval(30, 40)
    # Current implementation does not sort the points list in-place.
    assert combine_three_intervals(a, b, c) == Interval(10, 40)


def test_get_minimum_bounding_interval_uses_first_and_last_arguments_without_sorting():
    a = Interval(10, 20)
    b = Interval(1, 2)
    # Current implementation does not sort the points list in-place.
    assert get_minimum_bounding_interval(a, b) == Interval(10, 2)


def test_remove_returns_empty_list_when_other_contains_self():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner.remove(outer) == []


def test_remove_returns_original_interval_when_no_overlap():
    a = Interval(1, 5)
    b = Interval(10, 15)
    assert a.remove(b) == [a]


def test_remove_splits_interval_into_two_non_empty_segments():
    a = Interval(1, 10)
    b = Interval(3, 7)
    assert a.remove(b) == [Interval(1, 3), Interval(7, 10)]


def test_remove_when_overlap_touches_left_edge_returns_right_segment_only():
    a = Interval(1, 10)
    b = Interval(1, 4)
    assert a.remove(b) == [Interval(4, 10)]


def test_remove_when_overlap_touches_right_edge_returns_left_segment_only():
    a = Interval(1, 10)
    b = Interval(6, 10)
    assert a.remove(b) == [Interval(1, 6)]
