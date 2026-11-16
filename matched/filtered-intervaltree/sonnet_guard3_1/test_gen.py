from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize
    assert iv == (0, maxsize)


def test_interval_fields_and_unpacking():
    iv = Interval(3, 7)
    begin, end = iv
    assert (begin, end) == (3, 7)
    assert iv.begin == 3
    assert iv.end == 7


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 9)) == 7
    assert len(Interval(4, 4)) == 0


def test_interval_comparison_uses_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert Interval(1, 5) <= Interval(1, 9)
    assert Interval(1, 5) >= Interval(1, 9)


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(5, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_combine_intervals_disjoint():
    assert combine_intervals(Interval(1, 2), Interval(5, 8)) == Interval(1, 8)


def test_combine_intervals_reversed_order():
    assert combine_intervals(Interval(5, 8), Interval(1, 2)) == Interval(1, 8)


def test_combine_intervals_nested():
    assert combine_intervals(Interval(0, 10), Interval(3, 4)) == Interval(0, 10)
    assert combine_intervals(Interval(3, 4), Interval(0, 10)) == Interval(0, 10)


def test_combine_intervals_overlapping_and_identical():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)
    assert combine_intervals(Interval(2, 6), Interval(2, 6)) == Interval(2, 6)


def test_combine_intervals_negative_values():
    assert combine_intervals(Interval(-10, -5), Interval(-7, 3)) == Interval(-10, 3)


def test_combine_intervals_returns_interval_instance():
    result = combine_intervals(Interval(1, 2), Interval(3, 4))
    assert isinstance(result, Interval)


def test_add_matches_combine_intervals():
    pairs = [
        (Interval(1, 2), Interval(5, 8)),
        (Interval(5, 8), Interval(1, 2)),
        (Interval(0, 10), Interval(3, 4)),
        (Interval(-3, 4), Interval(2, 9)),
    ]
    for a, b in pairs:
        assert a + b == combine_intervals(a, b)


def test_get_minimum_bounding_interval_ordered_overlapping():
    result = get_minimum_bounding_interval(Interval(1, 5), Interval(2, 8))
    assert isinstance(result, Interval)
    assert result == Interval(1, 8)


def test_get_minimum_bounding_interval_does_not_sort_points():
    # The implementation discards the result of sorted(), so it returns
    # (a.begin, b.end) regardless of ordering.
    result = get_minimum_bounding_interval(Interval(5, 10), Interval(1, 3))
    assert result == Interval(5, 3)


def test_combine_three_intervals_ordered():
    result = combine_three_intervals(Interval(1, 3), Interval(2, 6), Interval(4, 9))
    assert isinstance(result, Interval)
    assert result == Interval(1, 9)


def test_combine_three_intervals_does_not_sort_points():
    # sorted() result is discarded: result is (a.begin, c.end).
    result = combine_three_intervals(Interval(5, 6), Interval(1, 2), Interval(3, 4))
    assert result == Interval(5, 4)
