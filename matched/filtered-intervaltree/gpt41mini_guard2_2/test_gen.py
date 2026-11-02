import pytest
from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_creation_defaults():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == 2**63 - 1 or interval.end > 10**9  # maxsize, platform dependent


def test_interval_creation_explicit_values():
    interval = Interval(10, 20)
    assert interval.begin == 10
    assert interval.end == 20


@pytest.mark.parametrize(
    "a, b, expected_lt",
    [
        (Interval(1, 5), Interval(2, 6), True),
        (Interval(5, 10), Interval(5, 10), False),
        (Interval(10, 20), Interval(5, 15), False),
    ],
)
def test_interval_lt(a, b, expected_lt):
    assert (a < b) is expected_lt


@pytest.mark.parametrize(
    "a, b, expected_le",
    [
        (Interval(1, 5), Interval(2, 6), True),
        (Interval(5, 10), Interval(5, 10), True),
        (Interval(10, 20), Interval(5, 15), False),
    ],
)
def test_interval_le(a, b, expected_le):
    assert (a <= b) is expected_le


@pytest.mark.parametrize(
    "a, b, expected_gt",
    [
        (Interval(1, 5), Interval(2, 6), False),
        (Interval(5, 10), Interval(5, 10), False),
        (Interval(10, 20), Interval(5, 15), True),
    ],
)
def test_interval_gt(a, b, expected_gt):
    assert (a > b) is expected_gt


@pytest.mark.parametrize(
    "a, b, expected_ge",
    [
        (Interval(1, 5), Interval(2, 6), False),
        (Interval(5, 10), Interval(5, 10), True),
        (Interval(10, 20), Interval(5, 15), True),
    ],
)
def test_interval_ge(a, b, expected_ge):
    assert (a >= b) is expected_ge


def test_interval_len_correct():
    interval = Interval(10, 20)
    assert len(interval) == 10


def test_interval_len_zero_length():
    interval = Interval(5, 5)
    assert len(interval) == 0


def test_interval_contains_true_and_false():
    a = Interval(10, 20)
    b = Interval(12, 18)
    c = Interval(5, 15)
    assert b in a
    assert not c in a
    # reflexive
    assert a in a


@pytest.mark.parametrize(
    "a, b, expected_overlaps",
    [
        (Interval(1, 5), Interval(4, 8), True),
        (Interval(1, 5), Interval(5, 10), False),
        (Interval(10, 20), Interval(15, 25), True),
        (Interval(10, 20), Interval(20, 30), False),
        (Interval(10, 20), Interval(5, 10), False),
        (Interval(10, 20), Interval(5, 15), True),
    ],
)
def test_interval_overlaps_method(a, b, expected_overlaps):
    assert a.overlaps(b) is expected_overlaps
    assert b.overlaps(a) is expected_overlaps


def test_interval_touches_true_and_false():
    a = Interval(10, 20)
    b = Interval(20, 30)
    c = Interval(5, 10)
    d = Interval(21, 30)
    assert a.touches(b)
    assert c.touches(a)
    assert not a.touches(d)
    assert not b.touches(c)


def test_combine_intervals_basic():
    a = Interval(1, 5)
    b = Interval(4, 10)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 10


def test_combine_intervals_non_overlapping():
    a = Interval(1, 3)
    b = Interval(10, 15)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 15


def test_combine_three_intervals_basic():
    a = Interval(1, 3)
    b = Interval(4, 10)
    c = Interval(9, 12)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 12


def test_combine_three_intervals_out_of_order():
    a = Interval(20, 25)
    b = Interval(5, 10)
    c = Interval(15, 16)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 5
    assert combined.end == 25


def test_get_minimum_bounding_interval_basic():
    a = Interval(3, 6)
    b = Interval(5, 9)
    bounding = get_minimum_bounding_interval(a, b)
    assert bounding.begin == 3
    assert bounding.end == 9


def test_get_minimum_bounding_interval_out_of_order():
    a = Interval(10, 15)
    b = Interval(1, 5)
    bounding = get_minimum_bounding_interval(a, b)
    assert bounding.begin == 1
    assert bounding.end == 15


def test_interval_add_returns_combined():
    a = Interval(1, 4)
    b = Interval(3, 10)
    c = a + b
    assert isinstance(c, Interval)
    assert c.begin == 1
    assert c.end == 10


def test_interval_add_non_overlapping():
    a = Interval(1, 3)
    b = Interval(10, 15)
    c = a + b
    assert c.begin == 1
    assert c.end == 15


def test_remove_interval_no_overlap_returns_self_list():
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]


def test_remove_interval_contained_returns_empty():
    a = Interval(1, 10)
    b = Interval(0, 20)
    result = a.remove(b)
    assert result == []


def test_remove_interval_partial_overlap_left_side():
    a = Interval(1, 10)
    b = Interval(5, 15)
    result = a.remove(b)
    # Result intervals after removing overlap are [1,5) and None (right segment length 0 or less)
    assert len(result) == 1
    assert result[0].begin == 1
    assert result[0].end == 5


def test_remove_interval_partial_overlap_right_side():
    a = Interval(5, 15)
    b = Interval(1, 10)
    result = a.remove(b)
    assert len(result) == 1
    assert result[0].begin == 10
    assert result[0].end == 15


def test_remove_interval_exact_overlap_no_segments():
    a = Interval(5, 10)
    b = Interval(5, 10)
    result = a.remove(b)
    assert result == []


def test_remove_interval_overlap_both_sides():
    a = Interval(1, 20)
    b = Interval(5, 15)
    result = a.remove(b)
    assert len(result) == 2
    assert result[0] == Interval(1, 5)
    assert result[1] == Interval(15, 20)
