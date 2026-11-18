import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_new_defaults_to_empty_range():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == sys.maxsize


def test_interval_equality_and_tuple_behavior():
    interval = Interval(1, 5)
    assert interval == (1, 5)
    assert interval == Interval(1, 5)
    assert interval != Interval(1, 6)


def test_interval_ordering_uses_begin_only():
    assert Interval(1, 10) < Interval(2, 3)
    assert Interval(1, 10) <= Interval(1, 0)
    assert Interval(3, 4) > Interval(2, 100)
    assert Interval(3, 4) >= Interval(3, 0)


def test_len_returns_difference_between_end_and_begin():
    assert len(Interval(2, 7)) == 5


def test_contains_true_when_other_is_fully_inside():
    outer = Interval(1, 10)
    inner = Interval(3, 6)
    assert inner in outer
    assert not (outer in inner)


def test_contains_false_when_other_extends_beyond_bounds():
    outer = Interval(1, 10)
    not_contained = Interval(0, 10)
    assert not (not_contained in outer)


def test_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 9)
    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_touches_true_only_when_endpoints_match():
    a = Interval(1, 5)
    b = Interval(5, 8)
    c = Interval(2, 6)
    assert a.touches(b)
    assert b.touches(a)
    assert not a.touches(c)


def test_add_returns_bounding_interval():
    a = Interval(3, 7)
    b = Interval(1, 5)
    assert a + b == Interval(1, 7)
    assert combine_intervals(a, b) == Interval(1, 7)


def test_combine_intervals_is_commutative():
    a = Interval(10, 20)
    b = Interval(2, 8)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(2, 20)


def test_get_minimum_bounding_interval_uses_min_and_max_bounds():
    a = Interval(10, 20)
    b = Interval(2, 8)
    # The intended minimum bounding interval is [2, 20].
    # This test catches the current implementation bug that returns [10, 8].
    assert get_minimum_bounding_interval(a, b) == Interval(2, 20)


def test_combine_three_intervals_uses_min_and_max_bounds():
    a = Interval(10, 20)
    b = Interval(2, 8)
    c = Interval(15, 30)
    # The intended minimum bounding interval is [2, 30].
    # This test catches the current implementation bug that returns [10, 30].
    assert combine_three_intervals(a, b, c) == Interval(2, 30)


def test_get_overlap_returns_contained_interval_when_fully_contained():
    outer = Interval(1, 10)
    inner = Interval(3, 6)
    assert outer._get_overlap(inner) == inner


def test_get_overlap_for_left_overlap_returns_expected_segment():
    a = Interval(5, 10)
    b = Interval(2, 7)
    assert a._get_overlap(b) == Interval(5, 7)


def test_get_overlap_for_right_overlap_returns_expected_segment():
    a = Interval(5, 10)
    b = Interval(8, 15)
    assert a._get_overlap(b) == Interval(8, 10)


def test_remove_when_not_overlapping_returns_original_interval():
    a = Interval(1, 5)
    b = Interval(6, 9)
    assert a.remove(b) == [a]


def test_remove_when_other_fully_contains_self_returns_empty_list():
    a = Interval(3, 6)
    b = Interval(1, 10)
    assert a.remove(b) == []


def test_remove_splits_interval_into_two_segments_when_other_cuts_middle():
    a = Interval(1, 10)
    b = Interval(4, 7)
    assert a.remove(b) == [Interval(1, 4), Interval(7, 10)]


def test_remove_with_touching_interval_keeps_original_interval():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert a.remove(b) == [a]
