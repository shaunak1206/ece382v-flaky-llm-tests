from sys import maxsize

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
    assert i.end == maxsize


def test_interval_fields_and_unpacking():
    i = Interval(2, 7)
    begin, end = i
    assert (begin, end) == (2, 7)
    assert i.begin == 2
    assert i.end == 7


def test_interval_len_is_width():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(4, 4)) == 0


def test_interval_ordering_uses_begin():
    a = Interval(1, 10)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a <= Interval(1, 2)
    assert a >= Interval(1, 2)


def test_interval_contains_and_overlaps_and_touches():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(5, 11) not in outer
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))


def test_interval_add_combines():
    assert Interval(1, 5) + Interval(3, 8) == Interval(1, 8)
    assert isinstance(Interval(1, 5) + Interval(3, 8), Interval)


def test_interval_remove_variants():
    # disjoint -> unchanged
    assert Interval(0, 5).remove(Interval(10, 12)) == [Interval(0, 5)]
    # fully covered -> empty
    assert Interval(2, 3).remove(Interval(0, 10)) == []
    # hole in the middle
    assert Interval(0, 10).remove(Interval(3, 5)) == [Interval(0, 3), Interval(5, 10)]
    # partial overlap
    assert Interval(0, 5).remove(Interval(3, 8)) == [Interval(0, 3)]


def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 8)) == Interval(1, 8)


def test_combine_intervals_disjoint():
    assert combine_intervals(Interval(1, 2), Interval(10, 20)) == Interval(1, 20)


def test_combine_intervals_contained():
    assert combine_intervals(Interval(0, 10), Interval(2, 3)) == Interval(0, 10)
    assert combine_intervals(Interval(2, 3), Interval(0, 10)) == Interval(0, 10)


def test_combine_intervals_commutative_and_identical():
    a = Interval(4, 9)
    b = Interval(-3, 6)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(-3, 9)
    assert combine_intervals(a, a) == a


def test_combine_intervals_returns_interval_type():
    result = combine_intervals(Interval(1, 2), Interval(3, 4))
    assert isinstance(result, Interval)
    assert result.begin == 1
    assert result.end == 4


def test_get_minimum_bounding_interval_ordered_disjoint():
    assert get_minimum_bounding_interval(Interval(1, 3), Interval(5, 9)) == Interval(1, 9)


def test_get_minimum_bounding_interval_overlapping_and_identical():
    assert get_minimum_bounding_interval(Interval(1, 6), Interval(4, 9)) == Interval(1, 9)
    assert get_minimum_bounding_interval(Interval(2, 4), Interval(2, 4)) == Interval(2, 4)


def test_get_minimum_bounding_interval_returns_interval():
    result = get_minimum_bounding_interval(Interval(0, 1), Interval(2, 3))
    assert isinstance(result, Interval)


def test_combine_three_intervals_ordered_disjoint():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert result == Interval(1, 6)
    assert isinstance(result, Interval)


def test_combine_three_intervals_overlapping_and_identical():
    assert combine_three_intervals(
        Interval(0, 5), Interval(2, 7), Interval(4, 10)
    ) == Interval(0, 10)
    same = Interval(3, 8)
    assert combine_three_intervals(same, same, same) == same
