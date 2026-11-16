import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)
from sys import maxsize


# ---------- Interval ----------

def test_interval_defaults():
    i = Interval()
    assert i.begin == 0
    assert i.end == maxsize


def test_interval_attributes_and_positional_access():
    i = Interval(2, 7)
    assert i.begin == 2
    assert i.end == 7
    assert i[0] == 2
    assert i[1] == 7


def test_interval_keyword_construction():
    i = Interval(end=9, begin=4)
    assert (i.begin, i.end) == (4, 9)


def test_interval_is_immutable():
    i = Interval(1, 2)
    with pytest.raises(AttributeError):
        i.begin = 5


def test_interval_len_is_end_minus_begin():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(5, 5)) == 0


def test_interval_equality():
    assert Interval(1, 2) == Interval(1, 2)
    assert Interval(1, 2) != Interval(1, 3)


def test_interval_ordering_uses_begin_only():
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
    assert Interval(0, 4) in outer
    assert Interval(5, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_combines_to_bounding_interval():
    assert Interval(1, 3) + Interval(2, 8) == Interval(1, 8)
    assert Interval(5, 6) + Interval(1, 2) == Interval(1, 6)
    assert Interval(0, 10) + Interval(3, 4) == Interval(0, 10)


def test_interval_overlaps():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))
    assert Interval(0, 10).overlaps(Interval(2, 3))
    assert not Interval(0, 2).overlaps(Interval(5, 8))
    # merely touching intervals do not overlap
    assert not Interval(0, 5).overlaps(Interval(5, 8))


def test_interval_touches():
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))
    assert not Interval(0, 5).touches(Interval(3, 8))


def test_interval_remove_middle_splits_in_two():
    assert Interval(0, 10).remove(Interval(3, 5)) == [Interval(0, 3), Interval(5, 10)]


def test_interval_remove_partial_overlap_and_disjoint_and_covered():
    assert Interval(0, 5).remove(Interval(3, 10)) == [Interval(0, 3)]
    disjoint = Interval(0, 2)
    assert disjoint.remove(Interval(5, 8)) == [disjoint]
    assert Interval(2, 3).remove(Interval(0, 10)) == []


# ---------- combine_intervals ----------

def test_combine_intervals_disjoint_in_either_order():
    a, b = Interval(1, 2), Interval(5, 9)
    assert combine_intervals(a, b) == Interval(1, 9)
    assert combine_intervals(b, a) == Interval(1, 9)


def test_combine_intervals_nested_and_identical():
    assert combine_intervals(Interval(0, 10), Interval(3, 4)) == Interval(0, 10)
    assert combine_intervals(Interval(3, 4), Interval(0, 10)) == Interval(0, 10)
    assert combine_intervals(Interval(2, 6), Interval(2, 6)) == Interval(2, 6)


def test_combine_intervals_overlapping_returns_interval_instance():
    result = combine_intervals(Interval(1, 5), Interval(3, 8))
    assert isinstance(result, Interval)
    assert result == Interval(1, 8)


# ---------- get_minimum_bounding_interval ----------

def test_get_minimum_bounding_interval_ordered_inputs():
    assert get_minimum_bounding_interval(Interval(1, 3), Interval(2, 8)) == Interval(1, 8)
    assert get_minimum_bounding_interval(Interval(1, 2), Interval(5, 6)) == Interval(1, 6)


def test_get_minimum_bounding_interval_returns_interval():
    result = get_minimum_bounding_interval(Interval(0, 1), Interval(1, 2))
    assert isinstance(result, Interval)
    assert result == Interval(0, 2)


# ---------- combine_three_intervals ----------

def test_combine_three_intervals_ordered_inputs():
    a, b, c = Interval(1, 3), Interval(2, 5), Interval(4, 9)
    assert combine_three_intervals(a, b, c) == Interval(1, 9)


def test_combine_three_intervals_disjoint_ordered():
    a, b, c = Interval(0, 1), Interval(10, 11), Interval(20, 30)
    result = combine_three_intervals(a, b, c)
    assert isinstance(result, Interval)
    assert result == Interval(0, 30)
