"""Unit tests for intervaltree/interval.py (Interval, combine_intervals,
combine_three_intervals, get_minimum_bounding_interval)."""

from sys import maxsize

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
    get_overlap,
)


def test_default_interval_spans_zero_to_maxsize():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_fields_and_tuple_equality():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    assert iv == (2, 7)
    assert isinstance(iv, tuple)


def test_len_is_end_minus_begin():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(0, 1)) == 1
    assert len(Interval(4, 4)) == 0


def test_ordering_compares_begin_only():
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(5, 6) > Interval(2, 3)
    assert Interval(2, 3) <= Interval(2, 50)
    assert Interval(2, 3) >= Interval(2, 50)
    assert not Interval(2, 3) < Interval(2, 50)
    ordered = sorted([Interval(5, 6), Interval(1, 9), Interval(3, 4)])
    assert [iv.begin for iv in ordered] == [1, 3, 5]


def test_contains_is_inclusive_on_both_ends():
    outer = Interval(1, 10)
    assert Interval(3, 5) in outer
    assert Interval(1, 10) in outer
    assert Interval(1, 10) in Interval(1, 10)
    assert Interval(0, 5) not in outer
    assert Interval(5, 11) not in outer


def test_overlaps_partial_and_contained():
    assert Interval(1, 5).overlaps(Interval(3, 8))
    assert Interval(1, 10).overlaps(Interval(3, 4))
    assert Interval(2, 4).overlaps(Interval(2, 4))


def test_overlaps_is_symmetric():
    pairs = [
        ((1, 5), (3, 8)),
        ((1, 10), (3, 4)),
        ((1, 3), (3, 5)),
        ((5, 6), (1, 2)),
        ((2, 4), (2, 4)),
    ]
    for a, b in pairs:
        left = Interval(*a)
        right = Interval(*b)
        assert left.overlaps(right) == right.overlaps(left)


def test_overlaps_false_when_touching_or_disjoint():
    assert not Interval(1, 3).overlaps(Interval(3, 5))
    assert not Interval(3, 5).overlaps(Interval(1, 3))
    assert not Interval(1, 2).overlaps(Interval(5, 6))
    assert not Interval(5, 6).overlaps(Interval(1, 2))


def test_touches_only_when_endpoints_meet():
    assert Interval(1, 3).touches(Interval(3, 5))
    assert Interval(3, 5).touches(Interval(1, 3))
    assert not Interval(1, 3).touches(Interval(4, 5))
    assert not Interval(1, 5).touches(Interval(3, 8))


def test_add_returns_hull_of_both():
    result = Interval(1, 3) + Interval(2, 5)
    assert isinstance(result, Interval)
    assert result == Interval(1, 5)
    assert Interval(1, 3) + Interval(10, 12) == Interval(1, 12)


def test_remove_non_overlapping_returns_self():
    assert Interval(1, 3).remove(Interval(5, 7)) == [Interval(1, 3)]
    # Touching at an endpoint is not an overlap, so nothing is removed.
    assert Interval(1, 3).remove(Interval(3, 5)) == [Interval(1, 3)]


def test_remove_fully_covered_returns_empty():
    assert Interval(3, 5).remove(Interval(1, 10)) == []
    assert Interval(1, 5).remove(Interval(1, 5)) == []


def test_remove_middle_splits_into_two():
    assert Interval(1, 10).remove(Interval(3, 5)) == [
        Interval(1, 3),
        Interval(5, 10),
    ]


def test_remove_trims_left_side():
    assert Interval(3, 8).remove(Interval(1, 5)) == [Interval(5, 8)]


def test_remove_trims_right_side():
    assert Interval(1, 5).remove(Interval(3, 8)) == [Interval(1, 3)]


def test_get_overlap_returns_intersection_bounds():
    assert get_overlap(Interval(1, 10), Interval(3, 5)) == Interval(3, 5)
    assert get_overlap(Interval(1, 5), Interval(3, 8)) == Interval(3, 5)
    assert get_overlap(Interval(3, 8), Interval(1, 5)) == Interval(3, 5)
    assert Interval(1, 5)._get_overlap(Interval(3, 8)) == Interval(3, 5)


def test_combine_intervals_overlapping_and_disjoint():
    assert combine_intervals(Interval(1, 3), Interval(2, 5)) == Interval(1, 5)
    assert combine_intervals(Interval(2, 5), Interval(1, 3)) == Interval(1, 5)
    # Disjoint inputs produce the hull, which spans the gap between them.
    assert combine_intervals(Interval(1, 2), Interval(5, 6)) == Interval(1, 6)


def test_combine_intervals_containment_and_shared_endpoints():
    assert combine_intervals(Interval(1, 10), Interval(3, 4)) == Interval(1, 10)
    assert combine_intervals(Interval(3, 4), Interval(1, 10)) == Interval(1, 10)
    assert combine_intervals(Interval(1, 3), Interval(1, 8)) == Interval(1, 8)
    assert isinstance(combine_intervals(Interval(1, 3), Interval(1, 8)), Interval)


def test_combine_intervals_matches_add_operator():
    pairs = [
        (Interval(0, 4), Interval(2, 9)),
        (Interval(5, 6), Interval(1, 2)),
        (Interval(2, 2), Interval(7, 9)),
    ]
    for a, b in pairs:
        expected = a + b
        assert combine_intervals(a, b) == expected
        assert combine_intervals(b, a) == expected


def test_combine_three_intervals_ordered_inputs():
    a = Interval(1, 2)
    b = Interval(3, 4)
    c = Interval(5, 6)
    result = combine_three_intervals(a, b, c)
    assert isinstance(result, Interval)
    assert result == Interval(1, 6)

    # Overlapping inputs where the widest end belongs to the last interval.
    assert combine_three_intervals(
        Interval(1, 4), Interval(2, 6), Interval(5, 9)
    ) == Interval(1, 9)


@pytest.mark.xfail(
    strict=True,
    reason="source calls sorted(points) without assigning the result, "
           "so the outer bounds are not the true min and max",
)
def test_combine_three_intervals_unordered_inputs():
    # Expected hull is (1, 9). The source returns (4, 9).
    result = combine_three_intervals(
        Interval(4, 6), Interval(1, 2), Interval(3, 9)
    )
    assert result == Interval(1, 9)


def test_get_minimum_bounding_interval_ordered_inputs():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(5, 8))
    assert isinstance(result, Interval)
    assert result == Interval(1, 8)
    # Touching inputs still bound correctly.
    assert get_minimum_bounding_interval(
        Interval(1, 3), Interval(3, 8)
    ) == Interval(1, 8)


@pytest.mark.xfail(
    strict=True,
    reason="source calls sorted(points) without assigning the result, "
           "so the outer bounds are not the true min and max",
)
def test_get_minimum_bounding_interval_unordered_inputs():
    # Expected hull is (1, 6). The source returns (5, 2).
    result = get_minimum_bounding_interval(Interval(5, 6), Interval(1, 2))
    assert result == Interval(1, 6)
