import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_new_defaults_to_full_range():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == sys.maxsize


def test_interval_equality_and_tuple_behavior():
    iv = Interval(2, 5)
    assert iv == (2, 5)
    assert tuple(iv) == (2, 5)


def test_interval_len_returns_difference_between_end_and_begin():
    assert len(Interval(3, 10)) == 7


def test_interval_ordering_compares_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(2, 3) > Interval(1, 100)
    assert Interval(1, 5) <= Interval(1, 1)
    assert Interval(2, 5) >= Interval(2, 999)


def test_interval_contains_true_when_other_is_fully_inside():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_other_is_not_fully_inside():
    outer = Interval(1, 10)
    overlapping = Interval(8, 12)
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


def test_interval_touches_true_only_on_shared_endpoint():
    a = Interval(1, 5)
    b = Interval(5, 9)
    c = Interval(6, 10)
    assert a.touches(b)
    assert not a.touches(c)


def test_combine_intervals_returns_min_begin_and_max_end():
    a = Interval(3, 7)
    b = Interval(1, 10)
    assert combine_intervals(a, b) == Interval(1, 10)


def test_combine_intervals_is_commutative():
    a = Interval(4, 9)
    b = Interval(2, 6)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(2, 9)


def test_get_minimum_bounding_interval_uses_all_bounds():
    a = Interval(10, 20)
    b = Interval(3, 8)
    assert get_minimum_bounding_interval(a, b) == Interval(3, 20)


def test_get_minimum_bounding_interval_with_negative_and_reversed_bounds():
    a = Interval(7, 9)
    b = Interval(-5, 2)
    assert get_minimum_bounding_interval(a, b) == Interval(-5, 9)


def test_combine_three_intervals_uses_all_intervals_bounds():
    a = Interval(10, 11)
    b = Interval(3, 4)
    c = Interval(7, 20)
    assert combine_three_intervals(a, b, c) == Interval(3, 20)


def test_combine_three_intervals_handles_unsorted_inputs():
    a = Interval(30, 40)
    b = Interval(5, 6)
    c = Interval(12, 15)
    assert combine_three_intervals(a, b, c) == Interval(5, 40)
