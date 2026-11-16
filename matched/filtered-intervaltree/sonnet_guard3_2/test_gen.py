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
    assert len(iv) == maxsize


def test_interval_is_tuple_like():
    iv = Interval(2, 7)
    assert iv == (2, 7)
    assert tuple(iv) == (2, 7)
    b, e = iv
    assert (b, e) == (2, 7)


def test_interval_len():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(5, 5)) == 0


def test_interval_ordering_by_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)
    # equal begin, different end
    c = Interval(1, 5)
    assert a <= c and a >= c
    assert not (a < c) and not (a > c)


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_is_union_bounds():
    result = Interval(1, 5) + Interval(3, 9)
    assert result == Interval(1, 9)
    assert Interval(3, 9) + Interval(1, 5) == Interval(1, 9)
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)


def test_overlaps():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))
    assert Interval(0, 10).overlaps(Interval(2, 3))
    # touching endpoints do not overlap
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert not Interval(0, 2).overlaps(Interval(4, 6))


def test_touches():
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))
    assert not Interval(0, 5).touches(Interval(3, 8))


def test_remove_middle_splits_in_two():
    assert Interval(0, 10).remove(Interval(3, 5)) == [Interval(0, 3), Interval(5, 10)]


def test_remove_fully_covered_returns_empty():
    assert Interval(2, 5).remove(Interval(0, 10)) == []
    assert Interval(2, 5).remove(Interval(2, 5)) == []


def test_remove_partial_overlap_right_side():
    assert Interval(0, 5).remove(Interval(3, 8)) == [Interval(0, 3)]


def test_remove_partial_overlap_left_side():
    assert Interval(3, 8).remove(Interval(0, 5)) == [Interval(5, 8)]


def test_remove_disjoint_returns_self():
    iv = Interval(0, 2)
    assert iv.remove(Interval(5, 9)) == [iv]


def test_combine_intervals_overlapping_and_disjoint():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)
    assert combine_intervals(Interval(3, 9), Interval(1, 5)) == Interval(1, 9)
    assert combine_intervals(Interval(0, 2), Interval(10, 12)) == Interval(0, 12)
    assert combine_intervals(Interval(0, 20), Interval(5, 6)) == Interval(0, 20)


def test_combine_intervals_returns_interval_instance():
    result = combine_intervals(Interval(1, 2), Interval(3, 4))
    assert isinstance(result, Interval)
    assert (result.begin, result.end) == (1, 4)


def test_get_minimum_bounding_interval_ordered_inputs():
    # Inputs given in ascending order: result spans first begin to last end.
    result = get_minimum_bounding_interval(Interval(1, 4), Interval(6, 10))
    assert isinstance(result, Interval)
    assert result == Interval(1, 10)


def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(Interval(0, 2), Interval(3, 5), Interval(6, 9))
    assert isinstance(result, Interval)
    assert result == Interval(0, 9)
