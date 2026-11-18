import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_new_defaults_to_maxsize_end():
    iv = Interval(3)
    assert iv.begin == 3
    assert iv.end == sys.maxsize


def test_interval_len_is_end_minus_begin():
    iv = Interval(2, 7)
    assert len(iv) == 5


def test_interval_contains_true_for_nested_interval():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_for_non_nested_interval():
    outer = Interval(1, 10)
    other = Interval(0, 7)
    assert other not in outer


def test_combine_intervals_chooses_min_begin_and_max_end():
    a = Interval(5, 8)
    b = Interval(2, 10)
    result = combine_intervals(a, b)
    assert result == Interval(2, 10)


def test_combine_intervals_handles_reverse_order_arguments():
    a = Interval(12, 15)
    b = Interval(3, 9)
    result = combine_intervals(a, b)
    assert result == Interval(3, 15)


def test_combine_three_intervals_is_expected_to_span_all_three():
    a = Interval(5, 8)
    b = Interval(2, 10)
    c = Interval(7, 12)
    result = combine_three_intervals(a, b, c)
    # The intended bounding interval is (2, 12)
    assert result == Interval(2, 12)


def test_combine_three_intervals_with_unsorted_inputs_spans_all_three():
    a = Interval(20, 30)
    b = Interval(-5, 0)
    c = Interval(7, 8)
    result = combine_three_intervals(a, b, c)
    assert result == Interval(-5, 30)


def test_get_minimum_bounding_interval_returns_bounds_for_sorted_inputs():
    a = Interval(1, 4)
    b = Interval(3, 9)
    result = get_minimum_bounding_interval(a, b)
    assert result == Interval(1, 9)


def test_get_minimum_bounding_interval_exposes_sorting_bug():
    a = Interval(10, 20)
    b = Interval(1, 2)
    result = get_minimum_bounding_interval(a, b)
    # Because sorted(points) is not assigned, this currently returns (10, 2)
    assert result == Interval(1, 20)


def test_interval_add_returns_bounding_interval():
    a = Interval(4, 6)
    b = Interval(1, 10)
    assert a + b == Interval(1, 10)
    assert b + a == Interval(1, 10)


def test_interval_overlaps_detects_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 9)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_interval_overlaps_false_for_touching_only():
    a = Interval(1, 5)
    b = Interval(5, 9)
    assert not a.overlaps(b)
    assert a.touches(b)
    assert b.touches(a)


def test_interval_remove_when_other_is_contained_returns_empty_list():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    assert outer.remove(inner) == []


def test_interval_remove_when_no_overlap_returns_self():
    a = Interval(1, 4)
    b = Interval(6, 8)
    assert a.remove(b) == [a]


def test_interval_remove_splits_interval_on_overlap():
    a = Interval(1, 10)
    b = Interval(4, 7)
    assert a.remove(b) == [Interval(1, 4), Interval(7, 10)]
