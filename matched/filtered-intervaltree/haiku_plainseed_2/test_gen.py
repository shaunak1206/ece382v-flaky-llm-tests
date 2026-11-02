"""Unit tests for intervaltree.interval: Interval, combine_intervals,
combine_three_intervals and get_minimum_bounding_interval."""
import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_arguments():
    default = Interval()
    assert default.begin == 0
    assert default.end == sys.maxsize
    assert Interval(5) == (5, sys.maxsize)


def test_interval_fields_and_len():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    assert isinstance(iv, tuple)
    assert len(iv) == 5
    assert len(Interval(3, 10)) == 7
    assert len(Interval(4, 4)) == 0


def test_interval_ordering_compares_begin_only():
    assert Interval(1, 5) < Interval(2, 3)
    assert Interval(2, 3) > Interval(1, 5)
    assert Interval(1, 5) <= Interval(1, 9)
    assert Interval(1, 5) >= Interval(1, 9)
    assert not Interval(1, 5) < Interval(1, 9)
    assert not Interval(1, 5) > Interval(1, 9)


def test_interval_contains_is_inclusive():
    outer = Interval(0, 10)
    assert Interval(0, 10) in outer
    assert Interval(2, 5) in outer
    assert Interval(5, 11) not in outer
    assert Interval(-1, 3) not in outer


def test_interval_add_spans_and_is_commutative():
    assert Interval(1, 3) + Interval(2, 6) == (1, 6)
    assert Interval(5, 6) + Interval(0, 1) == (0, 6)
    assert Interval(0, 1) + Interval(8, 9) == (0, 9)
    a, b = Interval(1, 3), Interval(2, 6)
    assert a + b == b + a


def test_overlaps_partial_and_nested():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))
    assert Interval(0, 10).overlaps(Interval(3, 4))
    assert Interval(3, 4).overlaps(Interval(0, 10))
    assert Interval(0, 10).overlaps(Interval(0, 10))


def test_overlaps_false_for_touching_and_disjoint():
    assert not Interval(0, 5).overlaps(Interval(5, 10))
    assert not Interval(5, 10).overlaps(Interval(0, 5))
    assert not Interval(0, 2).overlaps(Interval(4, 6))
    assert not Interval(4, 6).overlaps(Interval(0, 2))


def test_touches():
    assert Interval(0, 5).touches(Interval(5, 9))
    assert Interval(5, 9).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 9))
    assert not Interval(0, 5).touches(Interval(3, 9))


def test_get_overlap_returns_inner_when_contained():
    assert Interval(0, 10)._get_overlap(Interval(2, 4)) == (2, 4)


def test_get_overlap_partial_overlap_both_orders():
    assert Interval(0, 5)._get_overlap(Interval(3, 8)) == (3, 5)
    assert Interval(3, 8)._get_overlap(Interval(0, 5)) == (3, 5)


def test_remove_middle_splits_into_two_pieces():
    result = Interval(0, 10).remove(Interval(3, 6))
    assert result == [Interval(0, 3), Interval(6, 10)]


def test_remove_disjoint_or_touching_returns_self():
    assert Interval(0, 5).remove(Interval(7, 9)) == [Interval(0, 5)]
    assert Interval(0, 5).remove(Interval(5, 9)) == [Interval(0, 5)]


def test_remove_fully_covered_returns_empty():
    assert Interval(2, 4).remove(Interval(0, 10)) == []
    assert Interval(0, 10).remove(Interval(0, 10)) == []


def test_remove_overlapping_left_edge_keeps_right_piece():
    assert Interval(0, 10).remove(Interval(-5, 3)) == [Interval(3, 10)]


def test_remove_overlapping_right_edge_keeps_left_piece():
    assert Interval(0, 10).remove(Interval(7, 20)) == [Interval(0, 7)]


def test_combine_intervals_disjoint_spans_gap():
    result = combine_intervals(Interval(0, 2), Interval(7, 9))
    assert isinstance(result, Interval)
    assert result == (0, 9)


def test_combine_intervals_nested_and_commutative():
    assert combine_intervals(Interval(2, 4), Interval(0, 10)) == (0, 10)
    assert combine_intervals(Interval(0, 10), Interval(2, 4)) == (0, 10)
    a, b = Interval(1, 5), Interval(3, 9)
    assert combine_intervals(a, b) == combine_intervals(b, a) == (1, 9)


def test_combine_three_intervals_disjoint_in_order():
    result = combine_three_intervals(Interval(0, 2), Interval(3, 5), Interval(6, 9))
    assert result == (0, 9)


def test_combine_three_intervals_when_outer_ends_are_extremes():
    result = combine_three_intervals(Interval(0, 4), Interval(2, 3), Interval(3, 9))
    assert result == (0, 9)


@pytest.mark.xfail(
    strict=False,
    raises=AssertionError,
    reason="combine_three_intervals discards the sorted() result and uses raw first/last points",
)
def test_combine_three_intervals_middle_interval_holds_extremes():
    result = combine_three_intervals(Interval(2, 4), Interval(0, 10), Interval(3, 5))
    assert result == (0, 10)


def test_get_minimum_bounding_interval_ordered_disjoint():
    result = get_minimum_bounding_interval(Interval(0, 2), Interval(5, 9))
    assert isinstance(result, Interval)
    assert result == (0, 9)


@pytest.mark.xfail(
    strict=False,
    raises=AssertionError,
    reason="get_minimum_bounding_interval discards the sorted() result and uses raw first/last points",
)
def test_get_minimum_bounding_interval_nested_returns_true_bounds():
    result = get_minimum_bounding_interval(Interval(2, 4), Interval(0, 10))
    assert result == (0, 10)


@pytest.mark.xfail(
    strict=False,
    raises=AssertionError,
    reason="get_minimum_bounding_interval result depends on argument order",
)
def test_get_minimum_bounding_interval_is_argument_order_independent():
    a, b = Interval(0, 2), Interval(5, 9)
    assert get_minimum_bounding_interval(a, b) == get_minimum_bounding_interval(b, a)
