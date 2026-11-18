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
    assert i.end > i.begin


def test_interval_len_is_end_minus_begin():
    i = Interval(3, 10)
    assert len(i) == 7


def test_interval_comparison_uses_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    c = Interval(1, 5)

    assert a < b
    assert b > a
    assert a <= c
    assert a >= c
    assert not (b < a)


def test_interval_contains_another_interval_when_fully_enclosed():
    outer = Interval(2, 8)
    inner = Interval(3, 6)

    assert inner in outer
    assert not (outer in inner)


def test_interval_contains_false_when_not_fully_enclosed():
    a = Interval(2, 8)
    b = Interval(1, 6)

    assert not (b in a)


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 10)

    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_when_touching_only():
    a = Interval(1, 5)
    b = Interval(5, 10)

    assert not a.overlaps(b)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_add_returns_union_of_bounds():
    a = Interval(1, 5)
    b = Interval(3, 10)

    result = a + b
    assert result == Interval(1, 10)


def test_combine_intervals_returns_union_of_bounds():
    a = Interval(7, 9)
    b = Interval(2, 6)

    result = combine_intervals(a, b)
    assert result == Interval(2, 9)


def test_get_minimum_bounding_interval_with_sorted_inputs():
    a = Interval(1, 4)
    b = Interval(6, 8)

    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(1, 8)


def test_combine_three_intervals_with_sorted_inputs():
    a = Interval(1, 2)
    b = Interval(3, 5)
    c = Interval(7, 9)

    result = combine_three_intervals(a, b, c)
    assert result == Interval(1, 9)


def test_remove_returns_original_when_no_overlap():
    a = Interval(1, 5)
    b = Interval(6, 8)

    assert a.remove(b) == [a]


def test_remove_returns_empty_when_other_contains_self():
    a = Interval(3, 5)
    b = Interval(1, 10)

    assert a.remove(b) == []


def test_remove_splits_interval_when_other_overlaps_middle():
    a = Interval(1, 10)
    b = Interval(4, 7)

    assert a.remove(b) == [Interval(1, 4), Interval(7, 10)]


def test_remove_trims_left_side_when_other_overlaps_left_edge():
    a = Interval(1, 10)
    b = Interval(0, 4)

    assert a.remove(b) == [Interval(4, 10)]


def test_remove_trims_right_side_when_other_overlaps_right_edge():
    a = Interval(1, 10)
    b = Interval(7, 15)

    assert a.remove(b) == [Interval(1, 7)]


def test_overlap_method_on_identical_intervals():
    a = Interval(2, 6)
    b = Interval(2, 6)

    assert a.overlaps(b)
    assert a._get_overlap(b) == b
