import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_default():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == 2**63 - 1 or interval.end > 1000000  # maxsize on 64bit


def test_interval_creation_with_values():
    interval = Interval(5, 10)
    assert interval.begin == 5
    assert interval.end == 10


def test_interval_lt_and_le():
    a = Interval(1, 3)
    b = Interval(2, 5)
    assert (a < b) is True
    assert (a <= b) is True
    assert (b < a) is False
    assert (b <= b) is True


def test_interval_gt_and_ge():
    a = Interval(1, 3)
    b = Interval(2, 5)
    assert (b > a) is True
    assert (b >= a) is True
    assert (a > b) is False
    assert (a >= a) is True


def test_interval_contains_true_and_false():
    a = Interval(1, 10)
    b = Interval(2, 5)
    c = Interval(0, 12)
    assert b in a
    assert not a in b
    assert not a in c
    assert c in c  # an interval should contain itself


def test_interval_len():
    a = Interval(5, 10)
    assert len(a) == 5
    b = Interval(0, 0)
    assert len(b) == 0
    c = Interval(10, 5)
    assert len(c) == -5  # technically possible, no validation in Interval


def test_interval_add():
    a = Interval(1, 5)
    b = Interval(3, 7)
    c = a + b
    assert c.begin == 1
    assert c.end == 7

    d = Interval(10, 12)
    e = d + a
    assert e.begin == 1
    assert e.end == 12


def test_combine_intervals():
    a = Interval(1, 4)
    b = Interval(3, 6)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 6

    c = Interval(5, 10)
    combined2 = combine_intervals(b, c)
    assert combined2.begin == 3
    assert combined2.end == 10


def test_combine_three_intervals():
    a = Interval(1, 2)
    b = Interval(3, 5)
    c = Interval(4, 6)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 6

    d = Interval(-1, 0)
    combined2 = combine_three_intervals(d, a, c)
    assert combined2.begin == -1
    assert combined2.end == 6


def test_get_minimum_bounding_interval():
    a = Interval(1, 5)
    b = Interval(4, 10)
    bounding = get_minimum_bounding_interval(a, b)
    assert bounding.begin == 1
    assert bounding.end == 10

    c = Interval(-3, 0)
    bounding2 = get_minimum_bounding_interval(c, a)
    assert bounding2.begin == -3
    assert bounding2.end == 5


def test_interval_overlaps_method_and_function():
    a = Interval(1, 5)
    b = Interval(4, 7)
    c = Interval(5, 10)
    d = Interval(10, 15)

    assert a.overlaps(b) is True
    assert a.overlaps(c) is False  # touching but no overlap
    assert a.overlaps(d) is False

    # test underlying function too
    from intervaltree.interval import interval_overlaps
    assert interval_overlaps(a, b) is True
    assert interval_overlaps(a, c) is False
    assert interval_overlaps(a, d) is False


def test_interval_touches():
    a = Interval(1, 5)
    b = Interval(5, 10)
    c = Interval(0, 1)
    d = Interval(6, 7)

    assert a.touches(b) is True
    assert a.touches(c) is True
    assert a.touches(d) is False


def test_interval_remove_no_overlap():
    a = Interval(1, 5)
    b = Interval(6, 10)
    removed = a.remove(b)
    # no overlap, so returns itself
    assert removed == [a]


def test_interval_remove_fully_contained():
    a = Interval(1, 10)
    b = Interval(0, 20)
    removed = a.remove(b)
    # a is fully contained in b, so removing returns empty list
    assert removed == []


def test_interval_remove_partial_overlap_left():
    a = Interval(1, 10)
    b = Interval(0, 5)
    removed = a.remove(b)
    # overlap is the segment from 1 to 5
    # returned segments: (before overlap) none, (after overlap) (5, 10)
    assert len(removed) == 1
    assert removed[0] == Interval(5, 10)


def test_interval_remove_partial_overlap_right():
    a = Interval(1, 10)
    b = Interval(5, 15)
    removed = a.remove(b)
    # overlap is from 5 to 10
    # returned segments: (before overlap) (1, 5), (after overlap) none
    assert len(removed) == 1
    assert removed[0] == Interval(1, 5)


def test_interval_remove_partial_overlap_middle():
    a = Interval(1, 10)
    b = Interval(4, 6)
    removed = a.remove(b)
    # overlap is from 4 to 6
    # returned segments: (1, 4) and (6, 10)
    assert len(removed) == 2
    assert Interval(1, 4) in removed
    assert Interval(6, 10) in removed


def test_interval_remove_exact_match():
    a = Interval(1, 10)
    b = Interval(1, 10)
    removed = a.remove(b)
    # fully contained, empty list expected
    assert removed == []


def test_get_overlap_returns_correct_interval():
    a = Interval(3, 8)
    b = Interval(5, 7)
    overlapped = a._get_overlap(b)
    # b is contained in a, should return b
    assert overlapped == b

    c = Interval(1, 4)
    overlap2 = a._get_overlap(c)
    assert overlap2.begin == 3
    assert overlap2.end == 4

    d = Interval(6, 10)
    overlap3 = a._get_overlap(d)
    assert overlap3.begin == 6
    assert overlap3.end == 8

    
def test_touching_intervals_do_not_overlap():
    a = Interval(1, 5)
    b = Interval(5, 10)
    assert a.overlaps(b) is False
    assert b.overlaps(a) is False
    assert a.touches(b) is True


def test_len_with_equal_begin_end():
    a = Interval(5, 5)
    assert len(a) == 0


def test_interval_contains_itself():
    a = Interval(7, 9)
    assert a in a
