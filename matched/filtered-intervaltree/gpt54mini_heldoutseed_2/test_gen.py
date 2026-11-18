import sys

import pytest

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_default_constructor_uses_full_range():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == sys.maxsize
    assert len(interval) == sys.maxsize


def test_interval_len_with_negative_length_is_supported():
    interval = Interval(5, 3)
    assert len(interval) == -2


def test_interval_ordering_compares_begin_only():
    a = Interval(1, 10)
    b = Interval(1, 2)
    c = Interval(2, 3)

    assert a <= b
    assert not a < b
    assert a < c
    assert c > a
    assert c >= a


def test_interval_contains_true_when_second_is_fully_inside_first():
    outer = Interval(2, 10)
    inner = Interval(4, 7)

    assert inner in outer
    assert outer.__contains__(inner) is True


def test_interval_contains_false_when_not_fully_contained():
    outer = Interval(2, 10)
    overlapping = Interval(1, 5)

    assert overlapping not in outer


def test_interval_overlaps_true_for_partial_overlap():
    a = Interval(1, 5)
    b = Interval(4, 9)

    assert a.overlaps(b) is True
    assert b.overlaps(a) is True


def test_interval_overlaps_false_for_touching_intervals():
    a = Interval(1, 5)
    b = Interval(5, 8)

    assert a.overlaps(b) is False
    assert b.overlaps(a) is False
    assert a.touches(b) is True


def test_interval_touches_false_for_separated_intervals():
    a = Interval(1, 5)
    b = Interval(6, 8)

    assert a.touches(b) is False


def test_interval_add_returns_bounding_interval():
    a = Interval(3, 8)
    b = Interval(1, 5)

    combined = a + b

    assert combined == Interval(1, 8)


def test_combine_intervals_returns_bounding_interval():
    a = Interval(3, 8)
    b = Interval(1, 5)

    combined = combine_intervals(a, b)

    assert combined == Interval(1, 8)


def test_get_minimum_bounding_interval_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 4)
    b = Interval(3, 10)

    combined = get_minimum_bounding_interval(a, b)

    assert combined == Interval(1, 10)


def test_combine_three_intervals_returns_bounding_interval_for_ordered_inputs():
    a = Interval(1, 2)
    b = Interval(3, 7)
    c = Interval(4, 9)

    combined = combine_three_intervals(a, b, c)

    assert combined == Interval(1, 9)


def test_get_overlap_returns_contained_interval_when_second_is_inside_first():
    a = Interval(1, 10)
    b = Interval(3, 6)

    assert a._get_overlap(b) == b


def test_get_overlap_returns_left_overlap_when_second_starts_before_first():
    a = Interval(5, 10)
    b = Interval(2, 8)

    assert a._get_overlap(b) == Interval(5, 8)


def test_get_overlap_returns_right_overlap_when_second_starts_after_first():
    a = Interval(2, 7)
    b = Interval(5, 12)

    assert a._get_overlap(b) == Interval(5, 7)


def test_remove_returns_empty_list_when_other_contains_self():
    a = Interval(3, 6)
    b = Interval(1, 10)

    assert a.remove(b) == []


def test_remove_returns_original_interval_when_no_overlap():
    a = Interval(1, 4)
    b = Interval(6, 9)

    assert a.remove(b) == [a]


def test_remove_splits_interval_when_partially_overlapped_from_left():
    a = Interval(5, 10)
    b = Interval(2, 7)

    assert a.remove(b) == [Interval(7, 10)]


def test_remove_splits_interval_when_partially_overlapped_from_right():
    a = Interval(5, 10)
    b = Interval(8, 12)

    assert a.remove(b) == [Interval(5, 8)]


def test_remove_returns_two_segments_when_other_is_strictly_inside():
    a = Interval(1, 10)
    b = Interval(3, 7)

    assert a.remove(b) == [Interval(1, 3), Interval(7, 10)]
