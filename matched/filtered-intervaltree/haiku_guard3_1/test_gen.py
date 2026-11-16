from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------------------
# Interval construction and basic attributes
# ---------------------------------------------------------------------------

def test_interval_default_arguments_span_zero_to_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize
    assert iv == Interval(0, maxsize)


def test_interval_exposes_begin_and_end_attributes():
    iv = Interval(3, 7)
    assert iv.begin == 3
    assert iv.end == 7


def test_interval_defines_no_instance_dict():
    # __slots__ = () means instances carry no per-instance __dict__.
    iv = Interval(1, 2)
    assert not hasattr(iv, "__dict__")


def test_interval_is_equal_to_plain_tuple_with_same_values():
    assert Interval(1, 5) == (1, 5)


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "begin, end, expected",
    [
        (0, 0, 0),
        (0, 5, 5),
        (-3, 4, 7),
        (10, 25, 15),
    ],
)
def test_len_is_end_minus_begin(begin, end, expected):
    assert len(Interval(begin, end)) == expected


# ---------------------------------------------------------------------------
# Ordering: comparisons use only the begin value
# ---------------------------------------------------------------------------

def test_rich_comparisons_use_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    same_begin = Interval(1, 2)

    assert a < b
    assert not a > b
    assert b > a
    assert not b < a

    assert a <= same_begin
    assert a >= same_begin
    assert not a < same_begin
    assert not a > same_begin


def test_sorting_orders_by_begin_and_is_stable_for_ties():
    items = [Interval(3, 9), Interval(1, 2), Interval(3, 4)]
    result = sorted(items)
    # Python's sort is stable, so the two begin=3 intervals keep input order.
    assert result == [Interval(1, 2), Interval(3, 9), Interval(3, 4)]


# ---------------------------------------------------------------------------
# Containment, overlap, touching
# ---------------------------------------------------------------------------

def test_contains_true_for_nested_interval():
    outer = Interval(0, 10)
    assert outer.begin <= Interval(2, 5).begin
    assert Interval(2, 5) in outer


def test_contains_false_for_partially_overlapping_interval():
    outer = Interval(0, 10)
    assert Interval(5, 15) not in outer


def test_contains_true_for_identical_interval():
    iv = Interval(4, 8)
    assert iv in iv


def test_overlaps_for_intersecting_intervals():
    a = Interval(0, 5)
    b = Interval(4, 10)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_overlaps_false_for_disjoint_intervals():
    assert not Interval(0, 2).overlaps(Interval(5, 7))


def test_overlaps_false_for_intervals_that_only_share_an_endpoint():
    # Intervals are half-open: [0, 5) and [5, 10) do not overlap.
    assert not Interval(0, 5).overlaps(Interval(5, 10))


def test_touches_true_when_endpoints_meet():
    assert Interval(0, 5).touches(Interval(5, 10))
    assert Interval(5, 10).touches(Interval(0, 5))


def test_touches_false_when_separated():
    assert not Interval(0, 4).touches(Interval(5, 10))


# ---------------------------------------------------------------------------
# Addition (union hull)
# ---------------------------------------------------------------------------

def test_add_returns_hull_of_two_overlapping_intervals():
    assert Interval(0, 3) + Interval(2, 8) == Interval(0, 8)


def test_add_is_symmetric():
    a = Interval(0, 3)
    b = Interval(2, 8)
    assert a + b == b + a


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

def test_remove_returns_empty_list_when_self_is_contained_in_other():
    assert Interval(2, 5).remove(Interval(0, 10)) == []


def test_remove_splits_interval_when_other_is_strictly_inside():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_trims_when_overlap_is_partial():
    result = Interval(0, 10).remove(Interval(5, 15))
    assert result == [Interval(0, 5)]


def test_remove_returns_self_when_no_overlap():
    original = Interval(0, 5)
    assert original.remove(Interval(7, 9)) == [original]


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_disjoint_ordered_inputs():
    assert combine_intervals(Interval(0, 2), Interval(5, 7)) == Interval(0, 7)


def test_combine_intervals_nested_returns_outer_bounds():
    assert combine_intervals(Interval(0, 10), Interval(2, 5)) == Interval(0, 10)


def test_combine_intervals_is_independent_of_argument_order():
    a = Interval(5, 10)
    b = Interval(0, 3)
    assert combine_intervals(a, b) == Interval(0, 10)
    assert combine_intervals(b, a) == Interval(0, 10)


def test_combine_intervals_returns_interval_instance():
    result = combine_intervals(Interval(1, 2), Interval(3, 4))
    assert isinstance(result, Interval)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_get_minimum_bounding_interval_overlapping_ordered_inputs():
    assert get_minimum_bounding_interval(Interval(0, 4), Interval(2, 9)) == Interval(0, 9)


def test_get_minimum_bounding_interval_disjoint_ordered_inputs():
    assert get_minimum_bounding_interval(Interval(0, 1), Interval(5, 7)) == Interval(0, 7)


@pytest.mark.xfail(
    strict=True,
    reason="Known bug: the sorted() result is discarded, so the function "
           "returns (a.begin, b.end) instead of the true bounding interval "
           "when interval_b lies before interval_a.",
)
def test_get_minimum_bounding_interval_reversed_inputs_bug():
    result = get_minimum_bounding_interval(Interval(5, 10), Interval(0, 3))
    assert result == Interval(0, 10)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_disjoint_ordered_inputs():
    result = combine_three_intervals(
        Interval(0, 2), Interval(3, 4), Interval(5, 8)
    )
    assert result == Interval(0, 8)


def test_combine_three_intervals_chained_overlap_ordered_inputs():
    result = combine_three_intervals(
        Interval(0, 3), Interval(2, 6), Interval(5, 9)
    )
    assert result == Interval(0, 9)


@pytest.mark.xfail(
    strict=True,
    reason="Known bug: the sorted() result is discarded, so the function "
           "does not compute the true bounding interval for unordered or "
           "nested inputs.",
)
def test_combine_three_intervals_nested_inputs_bug():
    result = combine_three_intervals(
        Interval(0, 10), Interval(2, 5), Interval(3, 4)
    )
    assert result == Interval(0, 10)
