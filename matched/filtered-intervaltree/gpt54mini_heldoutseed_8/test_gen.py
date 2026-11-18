import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_defaults_to_zero_to_maxsize():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == sys.maxsize
    assert len(interval) == sys.maxsize


def test_interval_new_accepts_explicit_bounds():
    interval = Interval(3, 8)
    assert interval.begin == 3
    assert interval.end == 8
    assert len(interval) == 5


def test_interval_comparison_uses_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a


def test_interval_contains_true_when_second_is_fully_inside_first():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_not_fully_contained():
    outer = Interval(1, 10)
    not_inside = Interval(0, 7)
    assert not (not_inside in outer)


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True


def test_interval_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 8)
    assert a.overlaps(b) is False
    assert a.touches(b) is True


def test_interval_overlaps_false_for_disjoint_intervals():
    a = Interval(1, 3)
    b = Interval(4, 6)
    assert a.overlaps(b) is False
    assert b.overlaps(a) is False


def test_interval_add_returns_bounding_interval():
    a = Interval(1, 4)
    b = Interval(3, 10)
    result = a + b
    assert result == Interval(1, 10)


def test_combine_intervals_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 4)
    b = Interval(3, 10)
    result = combine_intervals(a, b)
    assert result == Interval(1, 10)


def test_get_minimum_bounding_interval_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 4)
    b = Interval(3, 10)
    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(1, 10)


def test_combine_three_intervals_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 2)
    b = Interval(3, 4)
    c = Interval(5, 9)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(1, 9)


def test_get_overlap_when_second_is_contained_returns_second_interval():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert outer._get_overlap(inner) == inner


def test_get_overlap_when_second_starts_before_first_returns_expected_interval():
    a = Interval(5, 10)
    b = Interval(2, 8)
    assert a._get_overlap(b) == Interval(5, 8)


def test_get_overlap_when_second_starts_inside_first_returns_expected_interval():
    a = Interval(2, 8)
    b = Interval(5, 12)
    assert a._get_overlap(b) == Interval(5, 8)


def test_remove_non_overlapping_interval_returns_original():
    a = Interval(1, 5)
    b = Interval(6, 8)
    assert a.remove(b) == [a]


def test_remove_containing_interval_returns_empty_list():
    a = Interval(1, 5)
    b = Interval(0, 10)
    assert a.remove(b) == []


def test_remove_partial_overlap_on_left_keeps_right_segment():
    a = Interval(1, 10)
    b = Interval(0, 4)
    assert a.remove(b) == [Interval(4, 10)]


def test_remove_partial_overlap_on_right_keeps_left_segment():
    a = Interval(1, 10)
    b = Interval(6, 12)
    assert a.remove(b) == [Interval(1, 6)]


def test_remove_middle_overlap_splits_into_two_segments():
    a = Interval(1, 10)
    b = Interval(4, 6)
    assert a.remove(b) == [Interval(1, 4), Interval(6, 10)]
