import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_default_construction_spans_zero_to_maxsize():
    assert Interval() == (0, sys.maxsize)


def test_begin_and_end_fields_are_stored():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7


def test_len_is_end_minus_begin():
    assert len(Interval(3, 10)) == 7


def test_len_of_zero_length_interval_is_zero():
    assert len(Interval(4, 4)) == 0


def test_lt_compares_begin_only():
    assert Interval(1, 10) < Interval(2, 3)
    assert not Interval(2, 3) < Interval(2, 10)


def test_gt_compares_begin_only():
    assert Interval(5, 6) > Interval(1, 2)
    assert not Interval(1, 2) > Interval(1, 9)


def test_le_and_ge_true_for_equal_begins():
    a = Interval(3, 4)
    b = Interval(3, 99)
    assert a <= b
    assert a >= b
    assert not a < b
    assert not a > b


def test_contains_strictly_nested_interval():
    assert Interval(2, 5) in Interval(0, 10)


def test_contains_identical_interval():
    assert Interval(0, 10) in Interval(0, 10)


def test_does_not_contain_partial_overlap():
    assert Interval(8, 12) not in Interval(0, 10)


def test_add_returns_hull_of_disjoint_intervals():
    assert Interval(1, 3) + Interval(5, 8) == Interval(1, 8)


def test_add_with_nested_interval_returns_outer():
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)


def test_overlaps_true_for_intersecting_intervals():
    assert Interval(1, 5).overlaps(Interval(4, 8))


def test_overlaps_false_when_only_touching():
    assert not Interval(1, 5).overlaps(Interval(5, 8))


def test_overlaps_false_when_disjoint():
    assert not Interval(1, 2).overlaps(Interval(5, 8))


def test_touches_when_end_meets_begin():
    assert Interval(1, 5).touches(Interval(5, 8))


def test_touches_false_when_separated():
    assert not Interval(1, 5).touches(Interval(6, 8))


def test_remove_returns_empty_when_self_contained_in_other():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_remove_returns_self_when_no_overlap():
    assert Interval(1, 2).remove(Interval(5, 6)) == [Interval(1, 2)]


def test_remove_inner_interval_splits_into_two_segments():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_left_overlap_drops_empty_segment():
    result = Interval(0, 10).remove(Interval(-5, 4))
    assert result == [Interval(4, 10)]


def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)


def test_combine_intervals_nested_returns_outer_both_orders():
    outer = Interval(0, 10)
    inner = Interval(2, 3)
    assert combine_intervals(outer, inner) == Interval(0, 10)
    assert combine_intervals(inner, outer) == Interval(0, 10)


def test_get_minimum_bounding_interval_ordered_inputs():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(5, 8))
    assert result == Interval(1, 8)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted() result is discarded, so the bounds are not the true min/max",
)
def test_get_minimum_bounding_interval_nested_inputs():
    result = get_minimum_bounding_interval(Interval(0, 10), Interval(2, 3))
    assert result == Interval(0, 10)


def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(
        Interval(1, 2), Interval(3, 4), Interval(5, 6)
    )
    assert result == Interval(1, 6)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted() result is discarded, so the bounds are not the true min/max",
)
def test_combine_three_intervals_unordered_inputs():
    result = combine_three_intervals(
        Interval(5, 6), Interval(1, 2), Interval(3, 4)
    )
    assert result == Interval(1, 6)
