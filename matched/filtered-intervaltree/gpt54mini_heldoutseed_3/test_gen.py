import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_values():
    i = Interval()
    assert i.begin == 0
    assert i.end == sys.maxsize
    assert len(i) == sys.maxsize


def test_interval_equality_and_namedtuple_behavior():
    i1 = Interval(1, 5)
    i2 = Interval(1, 5)
    i3 = Interval(2, 5)
    assert i1 == i2
    assert i1 != i3
    assert tuple(i1) == (1, 5)


def test_interval_ordering_uses_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(1, 100) <= Interval(1, 2)
    assert Interval(3, 4) > Interval(2, 100)
    assert Interval(3, 4) >= Interval(3, 100)


def test_interval_contains_interval_when_fully_enclosed():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert not (outer in inner)


def test_interval_contains_interval_when_not_enclosed():
    outer = Interval(1, 10)
    overlapping = Interval(8, 12)
    assert overlapping not in outer


def test_interval_overlaps_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert not a.overlaps(b)
    assert not b.overlaps(a)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_add_returns_bounding_interval():
    a = Interval(5, 10)
    b = Interval(1, 7)
    assert a + b == Interval(1, 10)
    assert b + a == Interval(1, 10)


def test_combine_intervals_returns_min_begin_and_max_end():
    a = Interval(5, 9)
    b = Interval(1, 12)
    assert combine_intervals(a, b) == Interval(1, 12)
    assert combine_intervals(b, a) == Interval(1, 12)


def test_get_minimum_bounding_interval_handles_unsorted_inputs():
    a = Interval(10, 20)
    b = Interval(1, 5)
    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(1, 20)


def test_combine_three_intervals_returns_overall_bounding_interval():
    a = Interval(10, 20)
    b = Interval(1, 5)
    c = Interval(7, 15)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(1, 20)


def test_combine_three_intervals_handles_non_monotonic_input_order():
    a = Interval(30, 40)
    b = Interval(5, 10)
    c = Interval(20, 25)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(5, 40)


def test_remove_returns_original_when_no_overlap():
    a = Interval(1, 5)
    b = Interval(6, 10)
    assert a.remove(b) == [a]


def test_remove_returns_empty_list_when_other_contains_self():
    a = Interval(3, 7)
    b = Interval(1, 10)
    assert a.remove(b) == []


def test_remove_splits_interval_when_other_overlaps_middle():
    a = Interval(1, 10)
    b = Interval(4, 7)
    assert a.remove(b) == [Interval(1, 4), Interval(7, 10)]


def test_remove_trims_left_side_when_overlap_starts_before_interval():
    a = Interval(5, 10)
    b = Interval(1, 7)
    assert a.remove(b) == [Interval(7, 10)]


def test_remove_trims_right_side_when_overlap_ends_after_interval():
    a = Interval(5, 10)
    b = Interval(8, 15)
    assert a.remove(b) == [Interval(5, 8)]
