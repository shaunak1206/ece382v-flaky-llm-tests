import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_defaults_to_full_range():
    i = Interval()
    assert i.begin == 0
    assert i.end == sys.maxsize
    assert len(i) == sys.maxsize


def test_interval_tuple_behavior_and_repr():
    i = Interval(2, 5)
    assert tuple(i) == (2, 5)
    assert i[0] == 2
    assert i[1] == 5
    assert repr(i) == "Interval(begin=2, end=5)"


def test_interval_comparison_uses_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    c = Interval(1, 2)

    assert a < b
    assert a <= c
    assert b > a
    assert b >= b
    assert not (b < a)


def test_interval_contains_true_for_fully_nested_interval():
    outer = Interval(1, 10)
    inner = Interval(3, 7)

    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_not_fully_nested():
    outer = Interval(1, 10)
    not_contained = Interval(0, 7)

    assert not_contained not in outer
    assert outer.__contains__(not_contained) is False


def test_interval_touches_detects_shared_boundary_left():
    a = Interval(1, 5)
    b = Interval(5, 9)

    assert a.touches(b)
    assert b.touches(a)


def test_interval_touches_detects_shared_boundary_right():
    a = Interval(10, 20)
    b = Interval(1, 10)

    assert a.touches(b)
    assert b.touches(a)


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 8)

    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_for_touching_only():
    a = Interval(1, 5)
    b = Interval(5, 8)

    assert not a.overlaps(b)
    assert not b.overlaps(a)


def test_interval_add_returns_mbounding_interval_like_union():
    a = Interval(1, 5)
    b = Interval(3, 10)

    result = a + b

    assert result == Interval(1, 10)
    assert isinstance(result, Interval)


def test_combine_intervals_returns_union_using_min_begin_and_max_end():
    a = Interval(10, 20)
    b = Interval(3, 15)

    result = combine_intervals(a, b)

    assert result == Interval(3, 20)


def test_combine_intervals_is_commutative_for_simple_inputs():
    a = Interval(2, 4)
    b = Interval(1, 10)

    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(1, 10)


def test_combine_three_intervals_returns_union_across_all_three():
    a = Interval(10, 20)
    b = Interval(3, 15)
    c = Interval(5, 30)

    result = combine_three_intervals(a, b, c)

    assert result == Interval(3, 30)


def test_combine_three_intervals_is_order_independent_for_simple_inputs():
    a = Interval(7, 9)
    b = Interval(1, 2)
    c = Interval(4, 8)

    assert combine_three_intervals(a, b, c) == Interval(1, 9)
    assert combine_three_intervals(c, a, b) == Interval(1, 9)


def test_get_minimum_bounding_interval_current_bug_returns_first_and_last_inputs():
    a = Interval(10, 20)
    b = Interval(1, 2)

    result = get_minimum_bounding_interval(a, b)

    # The current implementation does not use the sorted result.
    assert result == Interval(10, 2)


def test_get_minimum_bounding_interval_with_three_intervals_issue_is_visible():
    a = Interval(5, 6)
    b = Interval(1, 100)

    result = get_minimum_bounding_interval(a, b)

    assert result == Interval(5, 100)


def test_remove_returns_original_interval_when_no_overlap():
    a = Interval(1, 5)
    b = Interval(6, 8)

    assert a.remove(b) == [a]


def test_remove_returns_empty_list_when_other_is_fully_contained():
    a = Interval(1, 10)
    b = Interval(3, 7)

    assert a.remove(b) == []


def test_remove_splits_interval_when_overlap_on_right_side():
    a = Interval(1, 10)
    b = Interval(6, 15)

    result = a.remove(b)

    assert result == [Interval(1, 6)]


def test_remove_splits_interval_when_overlap_on_left_side():
    a = Interval(1, 10)
    b = Interval(-5, 4)

    result = a.remove(b)

    assert result == [Interval(4, 10)]


def test_remove_with_touching_interval_keeps_original():
    a = Interval(1, 5)
    b = Interval(5, 9)

    assert a.remove(b) == [a]
