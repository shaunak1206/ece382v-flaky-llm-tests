import pytest
from sys import maxsize

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


# ---------------------------------------------------------------------------
# Interval class
# ---------------------------------------------------------------------------

def test_default_interval_spans_zero_to_maxsize():
    assert Interval() == (0, maxsize)
    assert Interval(5) == (5, maxsize)


def test_interval_exposes_begin_and_end():
    iv = Interval(3, 9)
    assert iv.begin == 3
    assert iv.end == 9


def test_interval_is_immutable():
    iv = Interval(1, 4)
    with pytest.raises(AttributeError):
        iv.begin = 10
    with pytest.raises(AttributeError):
        iv.end = 10
    assert not hasattr(iv, "__dict__")


def test_interval_len_is_end_minus_begin():
    assert len(Interval(2, 7)) == 5
    assert len(Interval(3, 3)) == 0


def test_ordering_compares_begin_only():
    # The end value must not influence ordering.
    assert Interval(1, 100) < Interval(2, 3)
    assert Interval(2, 3) > Interval(1, 100)
    assert Interval(1, 2) <= Interval(1, 50)
    assert Interval(4, 5) >= Interval(3, 50)


def test_ordering_with_equal_begin():
    a = Interval(2, 3)
    b = Interval(2, 9)
    assert a <= b
    assert a >= b
    assert not (a < b)
    assert not (a > b)


def test_contains_nested_interval_only():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer  # an interval contains itself
    assert Interval(5, 15) not in outer
    assert Interval(5, 15) not in Interval(0, 10) or True  # explicit check below
    assert not outer.__contains__(Interval(-1, 5))


def test_add_overlapping_intervals_returns_union():
    assert Interval(1, 4) + Interval(3, 8) == Interval(1, 8)
    assert Interval(3, 8) + Interval(1, 4) == Interval(1, 8)


def test_add_nested_intervals_returns_outer():
    assert Interval(0, 10) + Interval(2, 3) == Interval(0, 10)


def test_add_disjoint_intervals_spans_gap():
    result = Interval(0, 2) + Interval(5, 7)
    assert isinstance(result, Interval)
    assert result == Interval(0, 7)


def test_overlaps_true_when_intersecting():
    a = Interval(0, 5)
    b = Interval(3, 8)
    assert a.overlaps(b)
    assert b.overlaps(a)


def test_overlaps_false_for_touching_intervals():
    # Sharing only an endpoint is not an overlap.
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert not Interval(5, 8).overlaps(Interval(0, 5))


def test_overlaps_false_for_disjoint_intervals():
    assert not Interval(0, 2).overlaps(Interval(5, 8))
    assert not Interval(5, 8).overlaps(Interval(0, 2))


def test_touches_when_endpoints_meet():
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))


def test_touches_false_when_separated_by_gap():
    assert not Interval(0, 4).touches(Interval(5, 8))
    assert not Interval(0, 5).touches(Interval(6, 8))


# ---------------------------------------------------------------------------
# combine_intervals
# ---------------------------------------------------------------------------

def test_combine_intervals_overlapping():
    result = combine_intervals(Interval(1, 4), Interval(3, 8))
    assert isinstance(result, Interval)
    assert result == Interval(1, 8)


def test_combine_intervals_nested_returns_outer_bounds():
    assert combine_intervals(Interval(0, 10), Interval(2, 3)) == Interval(0, 10)
    assert combine_intervals(Interval(2, 3), Interval(0, 10)) == Interval(0, 10)


def test_combine_intervals_disjoint_spans_gap():
    assert combine_intervals(Interval(5, 7), Interval(0, 2)) == Interval(0, 7)


# ---------------------------------------------------------------------------
# get_minimum_bounding_interval
# ---------------------------------------------------------------------------

def test_minimum_bounding_interval_with_ordered_inputs():
    # Works here only because the first listed begin is the minimum and
    # the last listed end is the maximum.
    result = get_minimum_bounding_interval(Interval(0, 5), Interval(2, 7))
    assert isinstance(result, Interval)
    assert result == Interval(0, 7)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted(points) result is discarded, so min/max are not computed.",
)
def test_minimum_bounding_interval_with_unordered_inputs():
    # Expected bounds are (0, 7); the buggy code returns (2, 5).
    result = get_minimum_bounding_interval(Interval(2, 7), Interval(0, 5))
    assert result == Interval(0, 7)


# ---------------------------------------------------------------------------
# combine_three_intervals
# ---------------------------------------------------------------------------

def test_combine_three_intervals_with_ordered_inputs():
    # Works here only because the first listed begin is the minimum and
    # the last listed end is the maximum.
    result = combine_three_intervals(
        Interval(0, 5), Interval(1, 6), Interval(2, 7)
    )
    assert isinstance(result, Interval)
    assert result == Interval(0, 7)


@pytest.mark.xfail(
    strict=True,
    reason="Bug: sorted(points) result is discarded, so min/max are not computed.",
)
def test_combine_three_intervals_with_unordered_inputs():
    # Expected bounds are (0, 9); the buggy code returns (4, 9).
    result = combine_three_intervals(
        Interval(4, 5), Interval(0, 1), Interval(2, 9)
    )
    assert result == Interval(0, 9)
