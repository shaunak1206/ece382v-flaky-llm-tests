import pytest

from sys import maxsize

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------------------
# Interval basics
# ---------------------------------------------------------------------------

def test_interval_stores_begin_and_end():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7


def test_interval_default_values():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 7)) == 5
    assert len(Interval(4, 4)) == 0


def test_interval_ordering_compares_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)
    # Same begin: <= and >= hold, strict comparisons do not
    c = Interval(1, 5)
    assert a <= c and a >= c
    assert not (a < c) and not (a > c)


def test_interval_equality_and_hash():
    assert Interval(1, 2) == Interval(1, 2)
    assert Interval(1, 2) != Interval(1, 3)
    assert hash(Interval(1, 2)) == hash(Interval(1, 2))
    assert len({Interval(1, 2), Interval(1, 2), Interval(2, 3)}) == 2


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(5, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_returns_union_bounds():
    assert Interval(1, 4) + Interval(3, 9) == Interval(1, 9)
    assert Interval(5, 6) + Interval(1, 2) == Interval(1, 6)


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 10)) == Interval(1, 10)


def test_combine_intervals_is_order_independent():
    a = Interval(1, 5)
    b = Interval(3, 10)
    assert combine_intervals(a, b) == combine_intervals(b, a)


def test_combine_intervals_disjoint_spans_gap():
    assert combine_intervals(Interval(0, 2), Interval(8, 10)) == Interval(0, 10)
    assert combine_intervals(Interval(8, 10), Interval(0, 2)) == Interval(0, 10)


def test_combine_intervals_nested_returns_outer_bounds():
    outer = Interval(0, 100)
    inner = Interval(10, 20)
    assert combine_intervals(outer, inner) == outer
    assert combine_intervals(inner, outer) == outer


def test_combine_intervals_identical_and_touching():
    assert combine_intervals(Interval(2, 4), Interval(2, 4)) == Interval(2, 4)
    assert combine_intervals(Interval(0, 5), Interval(5, 9)) == Interval(0, 9)


def test_combine_intervals_returns_interval_instance():
    result = combine_intervals(Interval(1, 2), Interval(3, 4))
    assert isinstance(result, Interval)


def test_combine_intervals_negative_values():
    assert combine_intervals(Interval(-10, -5), Interval(-7, 3)) == Interval(-10, 3)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_minimum_bounding_interval_overlapping_ordered():
    result = get_minimum_bounding_interval(Interval(1, 5), Interval(3, 10))
    assert isinstance(result, Interval)
    assert result == Interval(1, 10)


def test_minimum_bounding_interval_disjoint_ordered():
    result = get_minimum_bounding_interval(Interval(0, 2), Interval(8, 10))
    assert result == Interval(0, 10)


def test_minimum_bounding_interval_identical():
    result = get_minimum_bounding_interval(Interval(3, 7), Interval(3, 7))
    assert result == Interval(3, 7)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_ordered_chain():
    result = combine_three_intervals(Interval(1, 3), Interval(2, 5), Interval(4, 9))
    assert isinstance(result, Interval)
    assert result == Interval(1, 9)


def test_combine_three_intervals_identical():
    iv = Interval(2, 6)
    assert combine_three_intervals(iv, iv, iv) == Interval(2, 6)
