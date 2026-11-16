"""Unit tests for Interval, combine_intervals, combine_three_intervals and
get_minimum_bounding_interval in intervaltree/interval.py.

These tests use no randomness, no clock, no hashing of str/bytes, and no
module-level state changes, so they are deterministic in any order.

Two tests are marked xfail(strict=True). get_minimum_bounding_interval and
combine_three_intervals call sorted(points) but discard the returned list,
so they return (first point of interval_a, last point of interval_b) instead
of the true minimum and maximum. Those tests document the intended behaviour
and will flip to XPASS (a failure under strict mode) once the bug is fixed.
"""
import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# --- Interval construction and basic accessors -----------------------------

def test_default_construction_spans_zero_to_maxsize():
    from sys import maxsize

    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_fields_are_begin_and_end_and_len_is_difference():
    iv = Interval(3, 8)
    assert iv.begin == 3
    assert iv.end == 8
    assert len(iv) == 5
    assert len(Interval(4, 4)) == 0


# --- Ordering operators (compare by begin only) ----------------------------

def test_lt_compares_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert not (Interval(3, 4) < Interval(2, 100))


def test_le_ge_and_lt_with_equal_begin():
    a = Interval(1, 5)
    b = Interval(1, 9)
    assert a <= b
    assert a >= b
    assert not (a < b)
    assert not (a > b)


def test_gt_compares_begin_only():
    assert Interval(5, 6) > Interval(2, 9)
    assert not (Interval(2, 9) > Interval(5, 6))


# --- Containment -----------------------------------------------------------

def test_contains_nested_interval():
    outer = Interval(0, 10)
    assert Interval(2, 4) in outer
    assert Interval(0, 10) in outer


def test_contains_rejects_partial_overlap():
    outer = Interval(0, 10)
    assert Interval(5, 12) not in outer
    assert Interval(-1, 3) not in outer


# --- Overlap and touching --------------------------------------------------

def test_overlaps_partial_intersection():
    assert Interval(0, 5).overlaps(Interval(3, 8))


def test_overlaps_is_symmetric_for_nested_intervals():
    outer = Interval(0, 10)
    inner = Interval(2, 4)
    assert outer.overlaps(inner)
    assert inner.overlaps(outer)


def test_touching_intervals_touch_but_do_not_overlap():
    a = Interval(0, 5)
    b = Interval(5, 8)
    assert a.touches(b)
    assert b.touches(a)
    assert not a.overlaps(b)


def test_disjoint_intervals_neither_overlap_nor_touch():
    a = Interval(0, 2)
    b = Interval(5, 7)
    assert not a.overlaps(b)
    assert not a.touches(b)


# --- Addition (convex hull) ------------------------------------------------

def test_add_returns_hull_of_both_intervals():
    assert Interval(0, 3) + Interval(2, 8) == Interval(0, 8)
    assert isinstance(Interval(0, 3) + Interval(2, 8), Interval)


# --- Interval.remove -------------------------------------------------------

def test_remove_contained_interval_returns_empty_list():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_remove_inner_interval_splits_into_two_pieces():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_partially_overlapping_keeps_left_piece():
    result = Interval(0, 10).remove(Interval(5, 15))
    assert result == [Interval(0, 5)]


def test_remove_disjoint_interval_returns_self():
    original = Interval(0, 5)
    result = original.remove(Interval(7, 9))
    assert result == [Interval(0, 5)]


def test_remove_touching_interval_returns_self():
    result = Interval(0, 5).remove(Interval(5, 8))
    assert result == [Interval(0, 5)]


# --- combine_intervals -----------------------------------------------------

def test_combine_intervals_disjoint_gives_full_hull():
    result = combine_intervals(Interval(0, 2), Interval(5, 7))
    assert result == Interval(0, 7)
    assert isinstance(result, Interval)


def test_combine_intervals_nested_is_order_independent():
    outer = Interval(0, 10)
    inner = Interval(2, 4)
    assert combine_intervals(outer, inner) == Interval(0, 10)
    assert combine_intervals(inner, outer) == Interval(0, 10)


# --- combine_three_intervals and get_minimum_bounding_interval -------------

def test_combine_three_intervals_with_ordered_inputs():
    a = Interval(0, 4)
    b = Interval(2, 6)
    c = Interval(5, 9)
    assert combine_three_intervals(a, b, c) == Interval(0, 9)


def test_get_minimum_bounding_interval_with_ordered_inputs():
    a = Interval(0, 4)
    b = Interval(2, 9)
    assert get_minimum_bounding_interval(a, b) == Interval(0, 9)


@pytest.mark.xfail(
    strict=True,
    reason="sorted(points) result is discarded, so the bound is not the true min/max",
)
def test_get_minimum_bounding_interval_with_reversed_inputs():
    a = Interval(2, 9)
    b = Interval(0, 4)
    assert get_minimum_bounding_interval(a, b) == Interval(0, 9)


@pytest.mark.xfail(
    strict=True,
    reason="sorted(points) result is discarded, so the bound is not the true min/max",
)
def test_combine_three_intervals_with_unordered_inputs():
    a = Interval(5, 6)
    b = Interval(0, 1)
    c = Interval(3, 4)
    assert combine_three_intervals(a, b, c) == Interval(0, 6)
