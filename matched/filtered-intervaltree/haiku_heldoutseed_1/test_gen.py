import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_default_construction_spans_zero_to_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == sys.maxsize


def test_begin_and_end_attributes():
    iv = Interval(3, 9)
    assert iv.begin == 3
    assert iv.end == 9


def test_equals_plain_tuple():
    assert Interval(1, 2) == (1, 2)
    assert hash(Interval(1, 2)) == hash((1, 2))


def test_len_returns_span_length():
    assert len(Interval(2, 7)) == 5
    assert len(Interval(4, 4)) == 0


def test_lt_compares_begin_only():
    # Ends are ignored by ordering; only begin matters.
    assert Interval(1, 10) < Interval(2, 3)
    assert not Interval(2, 3) < Interval(1, 10)
    assert Interval(1, 10) > Interval(0, 20)


def test_comparison_operators_use_begin():
    a = Interval(2, 5)
    b = Interval(2, 9)
    assert a <= b
    assert a >= b
    assert not a < b
    assert not a > b


def test_add_returns_union_of_overlapping_intervals():
    result = Interval(1, 4) + Interval(3, 8)
    assert result == Interval(1, 8)
    assert isinstance(result, Interval)


def test_add_spans_gap_between_disjoint_intervals():
    result = Interval(1, 2) + Interval(5, 6)
    assert result == Interval(1, 6)


def test_contains_nested_interval():
    outer = Interval(0, 10)
    inner = Interval(2, 5)
    assert inner in outer
    assert outer in outer


def test_contains_rejects_partial_overlap():
    outer = Interval(0, 10)
    assert Interval(5, 12) not in outer


def test_overlaps_true_for_partial_overlap_both_directions():
    a = Interval(0, 5)
    b = Interval(3, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_overlaps_false_when_only_touching():
    assert not Interval(1, 3).overlaps(Interval(3, 5))


def test_touches_at_shared_boundary():
    assert Interval(1, 3).touches(Interval(3, 5))
    assert Interval(1, 3).touches(Interval(0, 1))
    assert not Interval(1, 3).touches(Interval(2, 5))


def test_remove_fully_contained_returns_empty_list():
    assert Interval(2, 4).remove(Interval(1, 5)) == []


def test_remove_middle_portion_splits_interval():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_left_partial_overlap_keeps_right_part():
    result = Interval(0, 10).remove(Interval(-5, 4))
    assert result == [Interval(4, 10)]


def test_remove_right_partial_overlap_keeps_left_part():
    result = Interval(0, 10).remove(Interval(8, 15))
    assert result == [Interval(0, 8)]


def test_remove_disjoint_returns_self():
    result = Interval(0, 2).remove(Interval(5, 7))
    assert result == [Interval(0, 2)]


def test_remove_touching_returns_self():
    result = Interval(0, 5).remove(Interval(5, 8))
    assert result == [Interval(0, 5)]


def test_combine_intervals_spans_both_inputs():
    result = combine_intervals(Interval(1, 4), Interval(3, 8))
    assert result == Interval(1, 8)


def test_combine_intervals_nested_and_order_independent():
    outer = Interval(0, 10)
    inner = Interval(2, 5)
    assert combine_intervals(outer, inner) == Interval(0, 10)
    assert combine_intervals(inner, outer) == Interval(0, 10)


def test_combine_three_intervals_with_ascending_inputs():
    a = Interval(1, 3)
    b = Interval(2, 6)
    c = Interval(5, 9)
    assert combine_three_intervals(a, b, c) == Interval(1, 9)


def test_get_minimum_bounding_interval_overlapping_inputs():
    result = get_minimum_bounding_interval(Interval(1, 5), Interval(3, 8))
    assert result == Interval(1, 8)


@pytest.mark.xfail(
    strict=True,
    reason="get_minimum_bounding_interval discards the result of sorted(), "
           "so nested inputs produce a wrong bounding interval",
)
def test_get_minimum_bounding_interval_nested_inputs():
    result = get_minimum_bounding_interval(Interval(0, 10), Interval(2, 4))
    assert result == Interval(0, 10)


@pytest.mark.xfail(
    strict=True,
    reason="combine_three_intervals discards the result of sorted(), "
           "so unsorted endpoints produce a wrong bounding interval",
)
def test_combine_three_intervals_unsorted_inputs():
    a = Interval(5, 10)
    b = Interval(1, 3)
    c = Interval(6, 8)
    assert combine_three_intervals(a, b, c) == Interval(1, 10)
