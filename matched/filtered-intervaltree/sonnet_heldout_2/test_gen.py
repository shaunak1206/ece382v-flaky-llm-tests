from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_fields_and_equality():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    assert iv == Interval(2, 7)
    assert tuple(iv) == (2, 7)


def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


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
    assert Interval(5, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_combines_to_bounding_interval():
    assert Interval(1, 4) + Interval(3, 9) == Interval(1, 9)
    assert Interval(5, 6) + Interval(0, 2) == Interval(0, 6)
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)


def test_overlaps_true_for_partial_overlap_and_containment():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))
    assert Interval(0, 10).overlaps(Interval(2, 4))
    assert Interval(2, 4).overlaps(Interval(0, 10))


def test_overlaps_false_for_disjoint_and_touching():
    assert not Interval(0, 5).overlaps(Interval(6, 8))
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert not Interval(5, 8).overlaps(Interval(0, 5))


def test_touches():
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))
    assert not Interval(0, 5).touches(Interval(3, 8))


def test_remove_middle_splits_into_two():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_overlapping_right_side_leaves_left_part():
    result = Interval(0, 10).remove(Interval(5, 15))
    assert result == [Interval(0, 5)]


def test_remove_overlapping_left_side_leaves_right_part():
    result = Interval(5, 15).remove(Interval(0, 8))
    assert result == [Interval(8, 15)]


def test_remove_disjoint_returns_self_and_fully_covered_returns_empty():
    assert Interval(0, 3).remove(Interval(5, 9)) == [Interval(0, 3)]
    assert Interval(3, 5).remove(Interval(0, 10)) == []


def test_combine_intervals_overlapping_disjoint_and_contained():
    assert combine_intervals(Interval(1, 5), Interval(3, 8)) == Interval(1, 8)
    assert combine_intervals(Interval(1, 2), Interval(6, 9)) == Interval(1, 9)
    assert combine_intervals(Interval(0, 10), Interval(2, 3)) == Interval(0, 10)


def test_combine_intervals_is_commutative_and_handles_identical():
    a = Interval(4, 7)
    b = Interval(1, 5)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(1, 7)
    assert combine_intervals(a, a) == a


def test_get_minimum_bounding_interval_ordered_inputs():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(5, 8))
    assert isinstance(result, Interval)
    assert result == Interval(1, 8)


def test_get_minimum_bounding_interval_overlapping_and_identical():
    assert get_minimum_bounding_interval(Interval(1, 5), Interval(3, 8)) == Interval(1, 8)
    assert get_minimum_bounding_interval(Interval(2, 6), Interval(2, 6)) == Interval(2, 6)


def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert isinstance(result, Interval)
    assert result == Interval(1, 6)


def test_combine_three_intervals_overlapping_inputs():
    result = combine_three_intervals(Interval(0, 4), Interval(2, 7), Interval(5, 10))
    assert result == Interval(0, 10)
