import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)
from sys import maxsize


# ---------------------------------------------------------------------------
# Interval construction and basic behaviour
# ---------------------------------------------------------------------------

def test_interval_defaults_span_zero_to_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_with_only_begin_uses_maxsize_end():
    iv = Interval(5)
    assert iv.begin == 5
    assert iv.end == maxsize


def test_interval_is_tuple_like_and_hashable():
    iv = Interval(1, 4)
    assert iv == (1, 4)
    assert hash(iv) == hash((1, 4))
    assert isinstance(iv, tuple)


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 7)) == 5
    assert len(Interval(3, 3)) == 0


# ---------------------------------------------------------------------------
# Interval comparisons (ordering is by begin only)
# ---------------------------------------------------------------------------

def test_interval_ordering_uses_begin_only():
    assert Interval(1, 10) < Interval(2, 3)
    assert Interval(2, 3) > Interval(1, 10)
    assert not (Interval(2, 3) < Interval(1, 10))


def test_interval_equal_begin_comparisons():
    a = Interval(4, 9)
    b = Interval(4, 1)
    assert a <= b
    assert a >= b
    assert not (a < b)
    assert not (a > b)


# ---------------------------------------------------------------------------
# Interval containment, overlap and touching
# ---------------------------------------------------------------------------

def test_contains_operator_for_nested_interval():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer


def test_contains_operator_includes_equal_bounds():
    outer = Interval(0, 10)
    assert Interval(0, 10) in outer


def test_contains_operator_rejects_partial_overlap():
    outer = Interval(0, 10)
    assert Interval(-1, 5) not in outer
    assert Interval(5, 11) not in outer


def test_overlaps_true_for_partial_overlap():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(0, 5))


def test_overlaps_false_for_touching_intervals():
    assert not Interval(0, 5).overlaps(Interval(5, 10))


def test_overlaps_false_for_disjoint_intervals():
    assert not Interval(0, 2).overlaps(Interval(4, 6))


def test_touches_detects_shared_endpoint_in_either_direction():
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))


def test_touches_false_when_gap_exists():
    assert not Interval(0, 5).touches(Interval(6, 8))


# ---------------------------------------------------------------------------
# Interval addition (union-like span)
# ---------------------------------------------------------------------------

def test_add_overlapping_intervals_gives_span():
    assert (Interval(0, 3) + Interval(2, 8)) == Interval(0, 8)


def test_add_disjoint_intervals_fills_gap():
    result = Interval(0, 2) + Interval(5, 6)
    assert isinstance(result, Interval)
    assert result == Interval(0, 6)


# ---------------------------------------------------------------------------
# Interval._get_overlap and remove
# ---------------------------------------------------------------------------

def test_get_overlap_partial_overlap_returns_shared_region():
    assert Interval(0, 10)._get_overlap(Interval(5, 15)) == Interval(5, 10)


def test_remove_returns_empty_when_self_is_contained():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_remove_returns_self_when_no_overlap():
    iv = Interval(0, 5)
    assert iv.remove(Interval(7, 9)) == [iv]


def test_remove_middle_overlap_splits_into_two_segments():
    result = Interval(0, 10).remove(Interval(3, 5))
    assert result == [Interval(0, 3), Interval(5, 10)]


def test_remove_left_overlap_leaves_right_segment_only():
    result = Interval(0, 10).remove(Interval(-5, 4))
    assert result == [Interval(4, 10)]


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_returns_min_begin_and_max_end():
    result = combine_intervals(Interval(0, 4), Interval(2, 9))
    assert isinstance(result, Interval)
    assert result == Interval(0, 9)


def test_combine_intervals_is_order_independent():
    a = Interval(1, 3)
    b = Interval(6, 8)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(1, 8)


def test_combine_intervals_nested_returns_outer():
    assert combine_intervals(Interval(0, 10), Interval(3, 4)) == Interval(0, 10)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(
        Interval(0, 2), Interval(3, 5), Interval(6, 9)
    )
    assert isinstance(result, Interval)
    assert result == Interval(0, 9)


def test_combine_three_intervals_with_identical_intervals():
    iv = Interval(4, 7)
    assert combine_three_intervals(iv, iv, iv) == Interval(4, 7)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_minimum_bounding_interval_ordered_disjoint_inputs():
    result = get_minimum_bounding_interval(Interval(0, 2), Interval(3, 5))
    assert isinstance(result, Interval)
    assert result == Interval(0, 5)


@pytest.mark.xfail(
    reason=(
        "Known bug: sorted(points) result is discarded, so the function "
        "returns Interval(points[0], points[-1]) on the unsorted list. "
        "For a containing interval_a it returns the wrong bounds."
    ),
    strict=False,
)
def test_minimum_bounding_interval_when_first_contains_second():
    result = get_minimum_bounding_interval(Interval(0, 10), Interval(2, 5))
    assert result == Interval(0, 10)


@pytest.mark.xfail(
    reason=(
        "Known bug: sorted(points) result is discarded; combine_three_intervals "
        "uses unsorted endpoints and can return wrong bounds."
    ),
    strict=False,
)
def test_combine_three_intervals_when_first_contains_others():
    result = combine_three_intervals(
        Interval(0, 20), Interval(3, 5), Interval(7, 9)
    )
    assert result == Interval(0, 20)
