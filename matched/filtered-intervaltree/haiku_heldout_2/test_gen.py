import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)

BUG_SORTED_DISCARDED = (
    "Known bug: get_minimum_bounding_interval/combine_three_intervals call "
    "sorted(points) without assigning the result, so points stay unsorted."
)


# --- Interval basics -------------------------------------------------------

def test_default_interval_spans_zero_to_maxsize():
    assert Interval() == Interval(0, sys.maxsize)


def test_interval_exposes_begin_and_end():
    iv = Interval(3, 7)
    assert iv.begin == 3
    assert iv.end == 7


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 9)) == 7
    assert len(Interval(4, 4)) == 0


def test_interval_ordering_compares_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)


def test_interval_ordering_equal_begin_is_le_and_ge():
    a = Interval(5, 6)
    b = Interval(5, 99)
    assert a <= b
    assert a >= b
    assert not (a < b)
    assert not (a > b)


def test_interval_contains_nested_interval():
    outer = Interval(0, 10)
    assert Interval(2, 4) in outer
    assert Interval(0, 10) in outer
    assert Interval(-1, 4) not in outer
    assert Interval(5, 11) not in outer


def test_interval_add_returns_bounding_interval():
    assert Interval(1, 3) + Interval(2, 8) == Interval(1, 8)
    assert Interval(1, 3) + Interval(5, 6) == Interval(1, 6)


def test_interval_overlaps_true_for_partial_overlap():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))


def test_interval_overlaps_false_when_only_touching():
    assert not Interval(0, 2).overlaps(Interval(2, 4))
    assert not Interval(2, 4).overlaps(Interval(0, 2))


def test_interval_touches_when_endpoints_meet():
    assert Interval(0, 2).touches(Interval(2, 4))
    assert Interval(2, 4).touches(Interval(0, 2))
    assert not Interval(0, 1).touches(Interval(2, 4))


def test_interval_remove_contained_returns_empty_list():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_interval_remove_middle_splits_into_two_segments():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_interval_remove_prefix_drops_empty_segment():
    result = Interval(0, 10).remove(Interval(0, 4))
    assert result == [Interval(4, 10)]


def test_interval_remove_disjoint_returns_self():
    iv = Interval(0, 3)
    assert iv.remove(Interval(5, 9)) == [iv]


# --- combine_intervals -----------------------------------------------------

def test_combine_intervals_disjoint_ordered():
    assert combine_intervals(Interval(1, 2), Interval(5, 9)) == Interval(1, 9)


def test_combine_intervals_reversed_argument_order():
    assert combine_intervals(Interval(5, 9), Interval(1, 2)) == Interval(1, 9)


def test_combine_intervals_nested_returns_outer():
    assert combine_intervals(Interval(0, 10), Interval(2, 4)) == Interval(0, 10)
    assert combine_intervals(Interval(2, 4), Interval(0, 10)) == Interval(0, 10)


def test_combine_intervals_identical_returns_same_span():
    iv = Interval(3, 7)
    assert combine_intervals(iv, iv) == Interval(3, 7)


# --- combine_three_intervals -----------------------------------------------

def test_combine_three_intervals_ordered_chain():
    a = Interval(0, 2)
    b = Interval(1, 5)
    c = Interval(4, 9)
    assert combine_three_intervals(a, b, c) == Interval(0, 9)


def test_combine_three_intervals_with_shared_endpoints():
    a = Interval(0, 3)
    b = Interval(3, 6)
    c = Interval(6, 10)
    assert combine_three_intervals(a, b, c) == Interval(0, 10)


@pytest.mark.xfail(strict=True, reason=BUG_SORTED_DISCARDED)
def test_combine_three_intervals_unordered_inputs():
    a = Interval(6, 10)
    b = Interval(0, 3)
    c = Interval(3, 6)
    assert combine_three_intervals(a, b, c) == Interval(0, 10)


# --- get_minimum_bounding_interval -----------------------------------------

def test_get_minimum_bounding_interval_disjoint_ordered():
    assert get_minimum_bounding_interval(Interval(1, 3), Interval(5, 8)) == Interval(1, 8)


@pytest.mark.xfail(strict=True, reason=BUG_SORTED_DISCARDED)
def test_get_minimum_bounding_interval_reversed_inputs():
    assert get_minimum_bounding_interval(Interval(5, 8), Interval(1, 3)) == Interval(1, 8)


@pytest.mark.xfail(strict=True, reason=BUG_SORTED_DISCARDED)
def test_get_minimum_bounding_interval_nested_inputs():
    assert get_minimum_bounding_interval(Interval(0, 10), Interval(2, 4)) == Interval(0, 10)
