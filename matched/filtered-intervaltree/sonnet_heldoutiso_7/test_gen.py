from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_values():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_fields_and_tuple_behaviour():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    assert iv == Interval(2, 7)
    assert iv == (2, 7)
    begin, end = iv
    assert (begin, end) == (2, 7)
    assert hash(iv) == hash(Interval(2, 7))


def test_interval_is_immutable():
    iv = Interval(1, 2)
    with pytest.raises(AttributeError):
        iv.begin = 5


def test_interval_len_is_end_minus_begin():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(5, 5)) == 0


def test_interval_comparisons_use_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert Interval(1, 5) <= Interval(1, 9)
    assert Interval(1, 5) >= Interval(1, 9)
    assert not (Interval(1, 5) < Interval(1, 9))


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_overlaps_and_touches():
    a = Interval(0, 5)
    assert a.overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(a)
    assert not a.overlaps(Interval(5, 8))
    assert not a.overlaps(Interval(10, 12))
    assert a.touches(Interval(5, 8))
    assert Interval(5, 8).touches(a)
    assert not a.touches(Interval(6, 8))


def test_interval_add_overlapping():
    result = Interval(1, 5) + Interval(3, 9)
    assert result == Interval(1, 9)
    assert isinstance(result, Interval)


def test_interval_add_nested_and_disjoint():
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)
    assert Interval(2, 3) + Interval(0, 10) == Interval(0, 10)
    assert Interval(1, 2) + Interval(8, 9) == Interval(1, 9)


def test_interval_remove_middle_splits_in_two():
    assert Interval(0, 10).remove(Interval(3, 5)) == [Interval(0, 3), Interval(5, 10)]


def test_interval_remove_disjoint_returns_self():
    iv = Interval(0, 5)
    assert iv.remove(Interval(10, 20)) == [iv]


def test_combine_intervals_overlapping():
    result = combine_intervals(Interval(1, 5), Interval(3, 9))
    assert result == Interval(1, 9)
    assert isinstance(result, Interval)


def test_combine_intervals_is_symmetric():
    a = Interval(1, 4)
    b = Interval(7, 12)
    assert combine_intervals(a, b) == Interval(1, 12)
    assert combine_intervals(b, a) == Interval(1, 12)


def test_combine_intervals_nested_and_identical():
    assert combine_intervals(Interval(0, 10), Interval(2, 3)) == Interval(0, 10)
    assert combine_intervals(Interval(2, 3), Interval(0, 10)) == Interval(0, 10)
    assert combine_intervals(Interval(4, 6), Interval(4, 6)) == Interval(4, 6)


def test_combine_intervals_negative_values_and_defaults():
    assert combine_intervals(Interval(-5, -1), Interval(-3, 2)) == Interval(-5, 2)
    assert combine_intervals(Interval(), Interval(5, 10)) == Interval(0, maxsize)


def test_get_minimum_bounding_interval_ordered_intervals():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(5, 10))
    assert result == Interval(1, 10)
    assert isinstance(result, Interval)


def test_get_minimum_bounding_interval_overlapping_ordered():
    assert get_minimum_bounding_interval(Interval(0, 6), Interval(4, 9)) == Interval(0, 9)


def test_get_minimum_bounding_interval_negative_values():
    assert get_minimum_bounding_interval(Interval(-10, -5), Interval(-2, 3)) == Interval(-10, 3)


def test_combine_three_intervals_ordered_disjoint():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert result == Interval(1, 6)
    assert isinstance(result, Interval)


def test_combine_three_intervals_overlapping_ordered():
    result = combine_three_intervals(Interval(0, 5), Interval(3, 8), Interval(6, 20))
    assert result == Interval(0, 20)
