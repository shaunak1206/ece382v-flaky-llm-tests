import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------------------
# Interval construction and basic behaviour
# ---------------------------------------------------------------------------

def test_default_interval_spans_zero_to_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == sys.maxsize


def test_construction_sets_begin_and_end():
    iv = Interval(3, 7)
    assert iv.begin == 3
    assert iv.end == 7
    assert tuple(iv) == (3, 7)


def test_len_is_end_minus_begin():
    assert len(Interval(2, 7)) == 5
    assert len(Interval(4, 4)) == 0


def test_add_returns_bounding_span_of_both_intervals():
    result = Interval(0, 3) + Interval(2, 8)
    assert isinstance(result, Interval)
    assert result == Interval(0, 8)


# ---------------------------------------------------------------------------
# Ordering operators compare only the begin value
# ---------------------------------------------------------------------------

def test_lt_compares_begin_only():
    assert Interval(1, 10) < Interval(2, 3)
    assert not Interval(2, 3) < Interval(2, 10)
    assert not Interval(3, 4) < Interval(2, 10)


def test_le_gt_ge_compare_begin_only():
    assert Interval(2, 3) <= Interval(2, 10)
    assert Interval(2, 3) >= Interval(2, 10)
    assert Interval(5, 6) > Interval(2, 10)
    assert Interval(5, 6) >= Interval(5, 99)
    assert not Interval(1, 6) > Interval(2, 3)
    assert not Interval(1, 6) >= Interval(2, 3)


# ---------------------------------------------------------------------------
# Containment, overlap and touching
# ---------------------------------------------------------------------------

def test_contains_nested_interval():
    outer = Interval(0, 10)
    inner = Interval(2, 4)
    assert inner in outer
    assert outer.__contains__(inner)


def test_contains_returns_false_for_partial_overlap():
    assert Interval(5, 20) not in Interval(0, 10)


def test_overlaps_true_for_partial_overlap():
    a = Interval(0, 5)
    b = Interval(3, 9)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_touching_intervals_do_not_overlap():
    assert not Interval(0, 3).overlaps(Interval(3, 5))


def test_touches_true_for_adjacent_intervals():
    assert Interval(0, 3).touches(Interval(3, 5))
    assert Interval(3, 5).touches(Interval(0, 3))


def test_touches_false_for_separated_intervals():
    assert not Interval(0, 3).touches(Interval(4, 5))


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

def test_remove_contained_interval_returns_empty_list():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_remove_partial_overlap_keeps_left_piece():
    result = Interval(0, 10).remove(Interval(5, 20))
    assert result == [Interval(0, 5)]


def test_remove_middle_splits_into_two_pieces():
    result = Interval(0, 10).remove(Interval(3, 6))
    assert result == [Interval(0, 3), Interval(6, 10)]


def test_remove_disjoint_returns_self():
    original = Interval(0, 3)
    assert original.remove(Interval(5, 8)) == [original]


# ---------------------------------------------------------------------------
# get_overlap (module-level helper used by remove)
# ---------------------------------------------------------------------------

def test_get_overlap_partial_returns_shared_span():
    overlap = Interval(0, 10)._get_overlap(Interval(5, 20))
    assert overlap == Interval(5, 10)


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(0, 5), Interval(3, 9)) == Interval(0, 9)


def test_combine_intervals_is_symmetric():
    a = Interval(3, 9)
    b = Interval(0, 5)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(0, 9)


def test_combine_intervals_nested_returns_outer():
    assert combine_intervals(Interval(0, 10), Interval(2, 4)) == Interval(0, 10)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_get_minimum_bounding_interval_ordered_disjoint():
    result = get_minimum_bounding_interval(Interval(0, 2), Interval(5, 8))
    assert isinstance(result, Interval)
    assert result == Interval(0, 8)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted() result is discarded, so points are not sorted; "
           "the begin of the first argument is used even when it is larger.",
)
def test_get_minimum_bounding_interval_reversed_order():
    result = get_minimum_bounding_interval(Interval(5, 8), Interval(0, 6))
    assert result == Interval(0, 8)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_ordered():
    result = combine_three_intervals(
        Interval(0, 2), Interval(3, 4), Interval(5, 9)
    )
    assert isinstance(result, Interval)
    assert result == Interval(0, 9)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted() result is discarded, so the span is taken from "
           "the first and last listed endpoints rather than the true min/max.",
)
def test_combine_three_intervals_unordered():
    result = combine_three_intervals(
        Interval(4, 6), Interval(1, 2), Interval(0, 9)
    )
    assert result == Interval(0, 9)
