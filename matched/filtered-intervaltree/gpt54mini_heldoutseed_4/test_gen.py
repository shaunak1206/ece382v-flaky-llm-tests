import sys
import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_values():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == sys.maxsize
    assert len(interval) == sys.maxsize


def test_interval_equality_and_tuple_behavior():
    interval = Interval(1, 5)
    assert interval == (1, 5)
    assert interval == Interval(1, 5)
    assert interval != Interval(1, 6)


def test_interval_ordering_uses_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(1, 100) <= Interval(1, 2)
    assert Interval(3, 4) > Interval(2, 100)
    assert Interval(3, 4) >= Interval(3, 0)


def test_interval_contains_true_when_other_is_fully_within():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_other_extends_beyond_bounds():
    outer = Interval(1, 10)
    not_contained = Interval(0, 7)
    assert not_contained not in outer


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True


def test_interval_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert a.overlaps(b) is False
    assert b.overlaps(a) is False
    assert a.touches(b) is True


def test_interval_overlaps_false_for_disjoint_intervals():
    a = Interval(1, 3)
    b = Interval(4, 6)
    assert a.overlaps(b) is False
    assert b.overlaps(a) is False


def test_interval_add_returns_bounding_interval():
    a = Interval(1, 5)
    b = Interval(3, 10)
    assert a + b == Interval(1, 10)
    assert b + a == Interval(1, 10)


def test_combine_intervals_returns_bounding_interval():
    a = Interval(2, 5)
    b = Interval(1, 10)
    assert combine_intervals(a, b) == Interval(1, 10)
    assert combine_intervals(b, a) == Interval(1, 10)


def test_get_minimum_bounding_interval_uses_original_first_and_last_points():
    a = Interval(10, 20)
    b = Interval(1, 2)
    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(10, 2)


def test_get_minimum_bounding_interval_with_sorted_inputs_matches_bounds():
    a = Interval(1, 2)
    b = Interval(10, 20)
    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(1, 20)


def test_combine_three_intervals_uses_first_and_last_points_without_sorting():
    a = Interval(10, 20)
    b = Interval(1, 2)
    c = Interval(3, 4)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(10, 4)


def test_combine_three_intervals_with_ascending_inputs_matches_bounds():
    a = Interval(1, 2)
    b = Interval(3, 4)
    c = Interval(5, 6)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(1, 6)


def test_remove_returns_empty_when_other_contains_self():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner.remove(outer) == []


def test_remove_returns_original_interval_when_no_overlap():
    a = Interval(1, 3)
    b = Interval(5, 7)
    assert a.remove(b) == [a]


def test_remove_splits_interval_when_overlapping_interior():
    a = Interval(1, 10)
    b = Interval(3, 7)
    assert a.remove(b) == [Interval(1, 3), Interval(7, 10)]


def test_remove_with_overlap_at_left_edge_returns_right_segment_only():
    a = Interval(1, 10)
    b = Interval(0, 5)
    assert a.remove(b) == [Interval(5, 10)]


def test_remove_with_overlap_at_right_edge_returns_left_segment_only():
    a = Interval(1, 10)
    b = Interval(7, 12)
    assert a.remove(b) == [Interval(1, 7)]
