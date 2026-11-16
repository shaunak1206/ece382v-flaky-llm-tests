import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# --- Interval construction and basic properties ---

def test_default_construction_spans_zero_to_maxsize():
    from sys import maxsize

    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_construction_stores_begin_and_end():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7


def test_len_is_end_minus_begin():
    assert len(Interval(3, 10)) == 7


def test_len_of_zero_length_interval_is_zero():
    assert len(Interval(4, 4)) == 0


# --- Comparisons (based on begin only) ---

@pytest.mark.parametrize(
    "a, b, lt, le, gt, ge",
    [
        (Interval(1, 5), Interval(2, 3), True, True, False, False),
        (Interval(2, 3), Interval(1, 5), False, False, True, True),
        (Interval(2, 9), Interval(2, 3), False, True, False, True),
    ],
)
def test_comparisons_use_begin_only(a, b, lt, le, gt, ge):
    assert (a < b) is lt
    assert (a <= b) is le
    assert (a > b) is gt
    assert (a >= b) is ge


# --- Containment ---

def test_contains_fully_inside_interval():
    outer = Interval(0, 10)
    inner = Interval(3, 6)
    assert inner in outer
    assert outer.__contains__(inner) is True


def test_contains_equal_interval():
    assert Interval(2, 5) in Interval(2, 5)


def test_does_not_contain_partial_overlap():
    assert Interval(5, 15) not in Interval(0, 10)


# --- Addition (union bounds) ---

def test_add_returns_bounding_union():
    result = Interval(1, 4) + Interval(3, 9)
    assert isinstance(result, Interval)
    assert (result.begin, result.end) == (1, 9)


# --- Overlap and touch semantics ---

def test_overlaps_true_for_intersecting_intervals_both_directions():
    a = Interval(0, 5)
    b = Interval(3, 8)
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True


def test_overlaps_false_for_touching_half_open_intervals():
    assert Interval(0, 3).overlaps(Interval(3, 6)) is False
    assert Interval(3, 6).overlaps(Interval(0, 3)) is False


def test_overlaps_false_for_disjoint_intervals():
    assert Interval(0, 2).overlaps(Interval(5, 9)) is False


def test_touches_when_end_meets_begin():
    assert Interval(0, 3).touches(Interval(3, 6)) is True
    assert Interval(3, 6).touches(Interval(0, 3)) is True


def test_touches_false_when_intervals_overlap_or_are_apart():
    assert Interval(0, 4).touches(Interval(2, 6)) is False
    assert Interval(0, 2).touches(Interval(5, 9)) is False


# --- remove() ---

def test_remove_disjoint_interval_returns_self():
    base = Interval(0, 5)
    result = base.remove(Interval(10, 20))
    assert result == [base]


def test_remove_fully_contained_interval_returns_empty_list():
    assert Interval(3, 6).remove(Interval(0, 10)) == []


def test_remove_interior_interval_splits_into_two_pieces():
    result = Interval(0, 10).remove(Interval(3, 6))
    assert [(iv.begin, iv.end) for iv in result] == [(0, 3), (6, 10)]


def test_remove_trims_left_edge():
    result = Interval(0, 10).remove(Interval(-5, 4))
    assert [(iv.begin, iv.end) for iv in result] == [(4, 10)]


def test_remove_trims_right_edge():
    result = Interval(0, 10).remove(Interval(7, 20))
    assert [(iv.begin, iv.end) for iv in result] == [(0, 7)]


# --- combine_intervals ---

def test_combine_intervals_returns_min_begin_and_max_end():
    result = combine_intervals(Interval(3, 8), Interval(1, 5))
    assert isinstance(result, Interval)
    assert (result.begin, result.end) == (1, 8)


def test_combine_intervals_with_nested_interval_keeps_outer_bounds():
    result = combine_intervals(Interval(0, 10), Interval(2, 4))
    assert (result.begin, result.end) == (0, 10)


# --- combine_three_intervals ---

def test_combine_three_intervals_with_ordered_extremes():
    a = Interval(0, 4)
    b = Interval(2, 6)
    c = Interval(5, 9)
    result = combine_three_intervals(a, b, c)
    assert isinstance(result, Interval)
    assert (result.begin, result.end) == (0, 9)


# --- get_minimum_bounding_interval ---

def test_get_minimum_bounding_interval_for_ordered_inputs():
    result = get_minimum_bounding_interval(Interval(1, 5), Interval(3, 8))
    assert (result.begin, result.end) == (1, 8)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted(points) result is discarded, so unsorted points give wrong bounds",
)
def test_get_minimum_bounding_interval_when_first_arg_starts_later():
    result = get_minimum_bounding_interval(Interval(3, 8), Interval(1, 5))
    assert (result.begin, result.end) == (1, 8)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted(points) result is discarded, so a wide interval's end is missed",
)
def test_get_minimum_bounding_interval_when_first_arg_contains_second():
    result = get_minimum_bounding_interval(Interval(1, 10), Interval(2, 3))
    assert (result.begin, result.end) == (1, 10)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted(points) result is discarded in combine_three_intervals",
)
def test_combine_three_intervals_when_middle_interval_is_widest():
    a = Interval(4, 5)
    b = Interval(0, 10)
    c = Interval(3, 6)
    result = combine_three_intervals(a, b, c)
    assert (result.begin, result.end) == (0, 10)
