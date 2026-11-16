import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# --- Interval construction and basic properties ---

def test_interval_stores_begin_and_end():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7


def test_interval_defaults_to_zero_and_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == sys.maxsize


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 7)) == 5


def test_interval_len_of_empty_span_is_zero():
    assert len(Interval(4, 4)) == 0


# --- Ordering comparisons (based on begin only) ---

def test_lt_compares_begin_only():
    assert Interval(1, 10) < Interval(2, 3)
    assert not Interval(2, 3) < Interval(2, 10)


def test_le_true_for_equal_begin():
    assert Interval(2, 3) <= Interval(2, 9)
    assert Interval(1, 3) <= Interval(2, 3)
    assert not Interval(3, 3) <= Interval(2, 3)


def test_gt_compares_begin_only():
    assert Interval(5, 6) > Interval(2, 100)
    assert not Interval(2, 100) > Interval(2, 3)


def test_ge_true_for_equal_begin():
    assert Interval(2, 3) >= Interval(2, 9)
    assert Interval(3, 3) >= Interval(2, 3)
    assert not Interval(1, 3) >= Interval(2, 3)


# --- Containment ---

def test_contains_nested_interval():
    assert Interval(2, 5) in Interval(0, 10)


def test_contains_itself():
    assert Interval(0, 10) in Interval(0, 10)


def test_does_not_contain_partially_overlapping_interval():
    assert Interval(5, 11) not in Interval(0, 10)


# --- Addition (bounding union) ---

def test_add_overlapping_intervals():
    assert Interval(1, 3) + Interval(2, 6) == Interval(1, 6)


def test_add_disjoint_intervals_spans_gap():
    assert Interval(1, 2) + Interval(5, 6) == Interval(1, 6)


# --- Overlap and touch ---

def test_overlaps_true_for_partial_overlap():
    assert Interval(1, 5).overlaps(Interval(4, 8))


def test_overlaps_is_symmetric():
    a, b = Interval(1, 5), Interval(4, 8)
    assert a.overlaps(b) == b.overlaps(a)


def test_overlaps_false_when_only_touching():
    assert not Interval(1, 5).overlaps(Interval(5, 8))


def test_overlaps_false_for_disjoint_intervals():
    assert not Interval(1, 2).overlaps(Interval(5, 8))


def test_touches_true_when_end_meets_begin():
    assert Interval(1, 5).touches(Interval(5, 8))


def test_touches_false_for_separated_intervals():
    assert not Interval(1, 5).touches(Interval(6, 8))


# --- Remove ---

def test_remove_non_overlapping_returns_self():
    iv = Interval(1, 3)
    assert iv.remove(Interval(5, 6)) == [Interval(1, 3)]


def test_remove_fully_containing_interval_returns_empty():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_remove_identical_interval_returns_empty():
    assert Interval(1, 4).remove(Interval(1, 4)) == []


def test_remove_right_partial_overlap_leaves_left_piece():
    assert Interval(0, 10).remove(Interval(5, 15)) == [Interval(0, 5)]


def test_remove_left_partial_overlap_leaves_right_piece():
    assert Interval(5, 10).remove(Interval(0, 7)) == [Interval(7, 10)]


def test_remove_middle_splits_into_two_pieces():
    assert Interval(0, 10).remove(Interval(3, 6)) == [
        Interval(0, 3),
        Interval(6, 10),
    ]


# --- combine_intervals ---

def test_combine_intervals_overlapping():
    result = combine_intervals(Interval(0, 5), Interval(3, 9))
    assert result == Interval(0, 9)
    assert isinstance(result, Interval)


def test_combine_intervals_disjoint_spans_gap():
    assert combine_intervals(Interval(1, 2), Interval(5, 6)) == Interval(1, 6)


def test_combine_intervals_when_one_contains_other():
    assert combine_intervals(Interval(0, 10), Interval(2, 3)) == Interval(0, 10)


# --- combine_three_intervals ---

def test_combine_three_intervals_chained_overlap():
    result = combine_three_intervals(Interval(0, 2), Interval(1, 5), Interval(4, 9))
    assert result == Interval(0, 9)
    assert isinstance(result, Interval)


@pytest.mark.xfail(
    strict=True,
    reason="get_minimum_bounding_interval/combine_three_intervals call sorted() "
           "without assigning the result, so unsorted endpoints give wrong bounds",
)
def test_combine_three_intervals_unsorted_input_bug():
    result = combine_three_intervals(Interval(5, 6), Interval(0, 1), Interval(2, 3))
    assert result == Interval(0, 6)


# --- get_minimum_bounding_interval ---

def test_get_minimum_bounding_interval_already_ordered():
    result = get_minimum_bounding_interval(Interval(0, 2), Interval(4, 9))
    assert result == Interval(0, 9)
    assert isinstance(result, Interval)


@pytest.mark.xfail(
    strict=True,
    reason="sorted() result is discarded, so unsorted endpoints give wrong bounds",
)
def test_get_minimum_bounding_interval_unsorted_input_bug():
    result = get_minimum_bounding_interval(Interval(4, 9), Interval(0, 2))
    assert result == Interval(0, 9)
