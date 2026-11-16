from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------- Interval

def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_attributes_and_tuple_behaviour():
    iv = Interval(3, 8)
    assert iv.begin == 3
    assert iv.end == 8
    assert iv == (3, 8)
    assert tuple(iv) == (3, 8)


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 9)) == 7
    assert len(Interval(5, 5)) == 0


def test_interval_comparisons_use_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)
    assert Interval(1, 5) <= Interval(1, 2)
    assert Interval(1, 5) >= Interval(1, 2)
    assert not (Interval(1, 5) < Interval(1, 2))
    assert not (Interval(1, 5) > Interval(1, 2))


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 3) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 4) in outer
    assert Interval(5, 11) not in outer
    assert outer not in Interval(2, 3)


def test_interval_add_combines_to_union_bounds():
    assert Interval(1, 5) + Interval(3, 9) == Interval(1, 9)
    assert Interval(3, 9) + Interval(1, 5) == Interval(1, 9)
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)
    assert isinstance(Interval(1, 2) + Interval(5, 6), Interval)


def test_overlaps_true_cases():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))
    assert Interval(0, 10).overlaps(Interval(2, 3))
    assert Interval(2, 3).overlaps(Interval(0, 10))


def test_overlaps_false_for_disjoint_and_touching():
    assert not Interval(0, 5).overlaps(Interval(6, 8))
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert not Interval(5, 8).overlaps(Interval(0, 5))


def test_touches():
    assert Interval(0, 5).touches(Interval(5, 10))
    assert Interval(5, 10).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))
    assert not Interval(0, 5).touches(Interval(3, 8))


def test_remove_middle_splits_into_two():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_overlap_on_right_side():
    result = Interval(0, 10).remove(Interval(5, 15))
    assert result == [Interval(0, 5)]


def test_remove_overlap_on_left_side():
    result = Interval(5, 15).remove(Interval(0, 10))
    assert result == [Interval(10, 15)]


def test_remove_prefix_leaves_remainder():
    result = Interval(0, 10).remove(Interval(0, 4))
    assert result == [Interval(4, 10)]


def test_remove_fully_covered_returns_empty():
    assert Interval(3, 5).remove(Interval(0, 10)) == []
    assert Interval(0, 10).remove(Interval(0, 10)) == []


def test_remove_disjoint_or_touching_returns_self():
    iv = Interval(0, 5)
    assert iv.remove(Interval(10, 20)) == [iv]
    assert iv.remove(Interval(5, 10)) == [iv]


# ------------------------------------------------------ combine functions

def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)


def test_combine_intervals_is_commutative_and_handles_containment():
    a = Interval(0, 10)
    b = Interval(2, 3)
    assert combine_intervals(a, b) == Interval(0, 10)
    assert combine_intervals(b, a) == Interval(0, 10)


def test_combine_intervals_disjoint_covers_gap():
    result = combine_intervals(Interval(1, 2), Interval(8, 9))
    assert result == Interval(1, 9)
    assert isinstance(result, Interval)


def test_combine_intervals_equal_intervals():
    assert combine_intervals(Interval(4, 6), Interval(4, 6)) == Interval(4, 6)


def test_get_minimum_bounding_interval_ordered_inputs():
    assert get_minimum_bounding_interval(Interval(1, 3), Interval(2, 10)) == Interval(1, 10)
    result = get_minimum_bounding_interval(Interval(1, 2), Interval(5, 8))
    assert result == Interval(1, 8)
    assert isinstance(result, Interval)


def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(Interval(1, 3), Interval(2, 5), Interval(4, 10))
    assert result == Interval(1, 10)
    assert isinstance(result, Interval)
