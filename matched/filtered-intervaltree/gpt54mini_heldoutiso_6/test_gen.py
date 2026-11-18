import sys
import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_defaults_to_full_range():
    i = Interval()
    assert i.begin == 0
    assert i.end == sys.maxsize
    assert len(i) == sys.maxsize


def test_interval_equality_and_tuple_behavior():
    i = Interval(1, 5)
    assert i == (1, 5)
    assert i == Interval(1, 5)
    assert repr(i).startswith("Interval(")


def test_interval_comparison_uses_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(1, 100) <= Interval(1, 1)
    assert Interval(3, 4) > Interval(2, 999)
    assert Interval(3, 4) >= Interval(3, 0)


def test_interval_contains_true_when_second_is_inside_first():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert not (outer in inner)


def test_interval_contains_false_when_only_partially_overlapping():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert b not in a
    assert a not in b


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_for_touching_only():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert not a.overlaps(b)
    assert not b.overlaps(a)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_add_returns_bounding_interval():
    a = Interval(2, 6)
    b = Interval(1, 10)
    assert a + b == Interval(1, 10)
    assert b + a == Interval(1, 10)


def test_combine_intervals_returns_bounding_interval_for_overlap():
    a = Interval(2, 6)
    b = Interval(4, 9)
    assert combine_intervals(a, b) == Interval(2, 9)
    assert combine_intervals(b, a) == Interval(2, 9)


def test_combine_intervals_returns_bounding_interval_for_disjoint_intervals():
    a = Interval(1, 3)
    b = Interval(10, 12)
    assert combine_intervals(a, b) == Interval(1, 12)
    assert combine_intervals(b, a) == Interval(1, 12)


def test_combine_three_intervals_returns_bounding_interval():
    a = Interval(5, 6)
    b = Interval(1, 2)
    c = Interval(10, 20)
    assert combine_three_intervals(a, b, c) == Interval(1, 20)
    assert combine_three_intervals(c, a, b) == Interval(1, 20)


def test_get_minimum_bounding_interval_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 4)
    b = Interval(2, 9)
    assert get_minimum_bounding_interval(a, b) == Interval(1, 9)


def test_get_minimum_bounding_interval_is_order_independent():
    a = Interval(10, 20)
    b = Interval(1, 5)
    # This documents the intended behavior; current implementation is order-dependent.
    assert get_minimum_bounding_interval(a, b) == Interval(1, 20)


def test_remove_returns_original_when_intervals_do_not_overlap():
    a = Interval(1, 5)
    b = Interval(6, 8)
    assert a.remove(b) == [a]


def test_remove_returns_empty_when_other_contains_self():
    a = Interval(3, 5)
    b = Interval(1, 10)
    assert a.remove(b) == []


def test_remove_splits_interval_when_overlap_in_middle():
    a = Interval(1, 10)
    b = Interval(4, 6)
    assert a.remove(b) == [Interval(1, 4), Interval(6, 10)]


def test_remove_trims_left_overlap():
    a = Interval(1, 10)
    b = Interval(1, 4)
    assert a.remove(b) == [Interval(4, 10)]


def test_remove_trims_right_overlap():
    a = Interval(1, 10)
    b = Interval(6, 10)
    assert a.remove(b) == [Interval(1, 6)]
