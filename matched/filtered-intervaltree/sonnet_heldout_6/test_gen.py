from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_construction_and_unpacking():
    iv = Interval(3, 9)
    assert iv.begin == 3
    assert iv.end == 9
    begin, end = iv
    assert (begin, end) == (3, 9)
    assert iv == (3, 9)


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 10)) == 8
    assert len(Interval(5, 5)) == 0


def test_interval_comparisons_use_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert Interval(1, 5) <= Interval(1, 2)
    assert Interval(1, 5) >= Interval(1, 2)
    assert not (Interval(1, 5) < Interval(1, 2))


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_returns_union_bounds():
    assert Interval(1, 3) + Interval(2, 5) == Interval(1, 5)
    assert Interval(5, 8) + Interval(1, 2) == Interval(1, 8)
    assert isinstance(Interval(1, 3) + Interval(2, 5), Interval)


def test_interval_overlaps_and_touches():
    assert Interval(1, 5).overlaps(Interval(4, 8))
    assert Interval(4, 8).overlaps(Interval(1, 5))
    assert not Interval(1, 5).overlaps(Interval(5, 8))
    assert not Interval(1, 2).overlaps(Interval(3, 4))
    assert Interval(1, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(1, 5))
    assert not Interval(1, 4).touches(Interval(5, 8))


def test_interval_remove_disjoint_returns_self():
    assert Interval(1, 3).remove(Interval(5, 8)) == [Interval(1, 3)]


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_overlapping():
    result = combine_intervals(Interval(1, 3), Interval(2, 5))
    assert result == Interval(1, 5)
    assert isinstance(result, Interval)


def test_combine_intervals_disjoint_covers_gap():
    assert combine_intervals(Interval(1, 2), Interval(8, 9)) == Interval(1, 9)


def test_combine_intervals_is_commutative():
    a, b = Interval(4, 7), Interval(1, 5)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(1, 7)


def test_combine_intervals_one_contains_other():
    outer, inner = Interval(0, 10), Interval(3, 4)
    assert combine_intervals(outer, inner) == outer
    assert combine_intervals(inner, outer) == outer


def test_combine_intervals_identical_and_negative_values():
    assert combine_intervals(Interval(2, 6), Interval(2, 6)) == Interval(2, 6)
    assert combine_intervals(Interval(-5, -1), Interval(-3, 2)) == Interval(-5, 2)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_get_minimum_bounding_interval_overlapping():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(2, 5))
    assert result == Interval(1, 5)
    assert isinstance(result, Interval)


def test_get_minimum_bounding_interval_disjoint():
    assert get_minimum_bounding_interval(Interval(1, 2), Interval(8, 9)) == Interval(1, 9)


def test_get_minimum_bounding_interval_identical():
    assert get_minimum_bounding_interval(Interval(4, 6), Interval(4, 6)) == Interval(4, 6)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_ordered_disjoint():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert result == Interval(1, 6)
    assert isinstance(result, Interval)


def test_combine_three_intervals_overlapping_chain():
    result = combine_three_intervals(Interval(0, 4), Interval(3, 7), Interval(6, 10))
    assert result == Interval(0, 10)


def test_combine_three_intervals_identical():
    iv = Interval(2, 9)
    assert combine_three_intervals(iv, iv, iv) == Interval(2, 9)
