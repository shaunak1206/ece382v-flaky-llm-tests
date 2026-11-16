import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)
from sys import maxsize


# ---------------------------------------------------------------------------
# Interval basics
# ---------------------------------------------------------------------------

def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_fields_and_tuple_behaviour():
    iv = Interval(3, 8)
    assert iv.begin == 3
    assert iv.end == 8
    assert iv[0] == 3 and iv[1] == 8
    begin, end = iv
    assert (begin, end) == (3, 8)
    assert iv == Interval(3, 8)
    assert iv != Interval(3, 9)


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 10)) == 8
    assert len(Interval(5, 5)) == 0
    assert len(Interval()) == maxsize


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 5) in outer
    assert Interval(5, 10) in outer


def test_interval_does_not_contain():
    outer = Interval(2, 8)
    assert Interval(1, 5) not in outer
    assert Interval(5, 9) not in outer
    assert Interval(0, 10) not in outer
    assert Interval(20, 30) not in outer


def test_interval_comparisons_use_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)
    assert Interval(1, 5) <= Interval(1, 9)
    assert Interval(1, 5) >= Interval(1, 9)


def test_interval_add_merges_to_bounding_interval():
    assert Interval(1, 5) + Interval(3, 9) == Interval(1, 9)
    assert Interval(5, 9) + Interval(1, 3) == Interval(1, 9)
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)
    assert isinstance(Interval(1, 2) + Interval(3, 4), Interval)


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_overlapping():
    result = combine_intervals(Interval(1, 5), Interval(3, 9))
    assert result == Interval(1, 9)
    assert isinstance(result, Interval)


def test_combine_intervals_disjoint_both_orders():
    a = Interval(1, 2)
    b = Interval(10, 20)
    assert combine_intervals(a, b) == Interval(1, 20)
    assert combine_intervals(b, a) == Interval(1, 20)


def test_combine_intervals_nested_both_orders():
    outer = Interval(0, 100)
    inner = Interval(10, 20)
    assert combine_intervals(outer, inner) == Interval(0, 100)
    assert combine_intervals(inner, outer) == Interval(0, 100)


def test_combine_intervals_identical_and_equal_bounds():
    a = Interval(4, 7)
    assert combine_intervals(a, Interval(4, 7)) == Interval(4, 7)
    assert combine_intervals(Interval(4, 7), Interval(4, 9)) == Interval(4, 9)
    assert combine_intervals(Interval(2, 7), Interval(4, 7)) == Interval(2, 7)


def test_combine_intervals_negative_and_float_values():
    assert combine_intervals(Interval(-10, -2), Interval(-5, 3)) == Interval(-10, 3)
    assert combine_intervals(Interval(0.5, 1.5), Interval(1.0, 2.5)) == Interval(0.5, 2.5)


def test_combine_intervals_matches_add_operator():
    a = Interval(3, 6)
    b = Interval(1, 4)
    assert combine_intervals(a, b) == a + b


def test_combine_intervals_does_not_mutate_inputs():
    a = Interval(1, 5)
    b = Interval(3, 9)
    combine_intervals(a, b)
    assert a == Interval(1, 5)
    assert b == Interval(3, 9)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_minimum_bounding_interval_disjoint_ordered():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(5, 9))
    assert result == Interval(1, 9)
    assert isinstance(result, Interval)


def test_minimum_bounding_interval_overlapping_ordered():
    result = get_minimum_bounding_interval(Interval(1, 5), Interval(3, 9))
    assert result == Interval(1, 9)


def test_minimum_bounding_interval_identical():
    result = get_minimum_bounding_interval(Interval(2, 6), Interval(2, 6))
    assert result == Interval(2, 6)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_disjoint_ordered():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert result == Interval(1, 6)
    assert isinstance(result, Interval)


def test_combine_three_intervals_overlapping_ordered():
    result = combine_three_intervals(Interval(1, 4), Interval(2, 5), Interval(3, 6))
    assert result == Interval(1, 6)


def test_combine_three_intervals_identical():
    iv = Interval(7, 9)
    assert combine_three_intervals(iv, iv, iv) == Interval(7, 9)
