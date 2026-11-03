import random
import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    random.seed(42)
    iv = Interval()
    assert iv.begin == 0
    # maxsize is platform-dependent but stable for a given build, test it's int and large
    assert isinstance(iv.end, int)
    assert iv.end > iv.begin


def test_interval_creation_custom():
    random.seed(42)
    iv = Interval(3, 7)
    assert iv.begin == 3
    assert iv.end == 7


def test_interval_len_positive():
    random.seed(42)
    iv = Interval(2, 10)
    assert len(iv) == 8


def test_interval_len_zero():
    random.seed(42)
    iv = Interval(5, 5)
    assert len(iv) == 0


def test_interval_len_negative_should_not_occur():
    random.seed(42)
    iv = Interval(10, 3)
    # Length can be negative, but test len returns (end - begin) without error
    assert len(iv) == -7


def test_combine_intervals_basic():
    random.seed(42)
    a = Interval(1, 4)
    b = Interval(3, 8)
    c = combine_intervals(a, b)
    assert c.begin == min(a.begin, b.begin)
    assert c.end == max(a.end, b.end)


def test_combine_intervals_disjoint():
    random.seed(42)
    a = Interval(1, 2)
    b = Interval(5, 7)
    c = combine_intervals(a, b)
    assert c.begin == 1
    assert c.end == 7


def test_combine_three_intervals_order_irrelevant():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(2, 6)
    c = Interval(4, 10)
    combined1 = combine_three_intervals(a, b, c)
    combined2 = combine_three_intervals(c, a, b)
    combined3 = combine_three_intervals(b, c, a)
    for combined in (combined1, combined2, combined3):
        assert combined.begin == 1
        assert combined.end == 10


def test_combine_three_intervals_negative_and_overlapping():
    random.seed(42)
    a = Interval(-5, 0)
    b = Interval(-3, 1)
    c = Interval(-10, -6)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == -10
    assert combined.end == 1


def test_get_minimum_bounding_interval_basic():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(4, 10)
    bounding = get_minimum_bounding_interval(a, b)
    assert bounding.begin == 1
    assert bounding.end == 10


def test_get_minimum_bounding_interval_order_irrelevant():
    random.seed(42)
    a = Interval(5, 15)
    b = Interval(3, 7)
    bounding1 = get_minimum_bounding_interval(a, b)
    bounding2 = get_minimum_bounding_interval(b, a)
    assert bounding1.begin == 3
    assert bounding1.end == 15
    assert bounding2.begin == 3
    assert bounding2.end == 15


def test_interval_lt_and_le_and_gt_and_ge():
    random.seed(42)
    a = Interval(1, 3)
    b = Interval(2, 4)
    # less than based on begin
    assert (a < b) is True
    assert (a <= b) is True
    assert (b > a) is True
    assert (b >= a) is True
    # equal begins
    c = Interval(1, 10)
    assert (a < c) is False
    assert (a <= c) is True
    assert (c > a) is False
    assert (c >= a) is True


def test_interval_contains_true_and_false():
    random.seed(42)
    a = Interval(1, 10)
    b = Interval(3, 7)
    c = Interval(0, 5)
    d = Interval(5, 15)
    assert b in a
    assert not a in b
    assert not c in b
    assert not d in a


def test_interval_addition_synergy():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(3, 7)
    c = a + b
    # same as combine_intervals
    assert c.begin == 1
    assert c.end == 7


def test_interval_overlaps_method_true_false():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(4, 7)
    c = Interval(5, 9)
    d = Interval(6, 10)
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True
    # touching intervals do not overlap (5 is end of a and begin of c)
    assert a.overlaps(c) is False
    assert c.overlaps(a) is False
    assert c.overlaps(d) is True


def test_interval_touches_true_false():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(5, 7)
    c = Interval(6, 10)
    d = Interval(0, 1)
    assert a.touches(b) is True
    assert b.touches(a) is True
    assert not a.touches(c)
    assert a.touches(d) is True
    assert not b.touches(d)


def test_get_overlap_enclosed():
    random.seed(42)
    a = Interval(1, 10)
    b = Interval(3, 7)
    overlap = a._get_overlap(b)
    # b is fully contained in a; overlap should be b
    assert overlap.begin == b.begin
    assert overlap.end == b.end


def test_get_overlap_partial_left():
    random.seed(42)
    a = Interval(3, 10)
    b = Interval(1, 5)
    overlap = a._get_overlap(b)
    # overlap should be from 3 to 5
    assert overlap.begin == 3
    assert overlap.end == 5


def test_get_overlap_partial_right():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(3, 10)
    overlap = a._get_overlap(b)
    # overlap should be from 3 to 5 (a.end)
    assert overlap.begin == 3
    assert overlap.end == 5


def test_remove_non_overlapping_returns_self():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]


def test_remove_fully_contained_returns_empty():
    random.seed(42)
    a = Interval(3, 5)
    b = Interval(1, 10)
    result = a.remove(b)
    assert result == []


def test_remove_partial_left_overlap():
    random.seed(42)
    a = Interval(1, 7)
    b = Interval(3, 5)
    # remove overlap from a
    result = a.remove(b)
    # should have two intervals: (1,3) and (5,7)
    assert len(result) == 2
    left, right = result
    assert left.begin == 1
    assert left.end == 3
    assert right.begin == 5
    assert right.end == 7


def test_remove_partial_touching_removes_overlap_only():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(5, 6)
    # touching intervals do not overlap
    result = a.remove(b)
    # nothing removed because no overlap
    assert result == [a]


def test_remove_partial_overlap_results_non_empty_segments():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(2, 4)
    result = a.remove(b)
    # segments: (1,2) and (4,5), both length 1
    assert len(result) == 2
    for segment in result:
        assert 0 < len(segment) <= 2  # length must be positive


def test_remove_partial_overlap_results_no_zero_length_intervals():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(1, 3)
    result = a.remove(b)
    # segments: (3,5) only since (1,1) segment length zero is excluded
    assert len(result) == 1
    seg = result[0]
    assert seg.begin == 3 and seg.end == 5


def test_remove_when_other_contains_self():
    random.seed(42)
    a = Interval(4, 6)
    b = Interval(1, 10)
    result = a.remove(b)
    # Since a in b, removal should yield empty list
    assert result == []


def test_remove_when_self_contains_other_and_overlap_false():
    random.seed(42)
    a = Interval(1, 5)
    b = Interval(6, 8)
    result = a.remove(b)
    # No overlap => returns self intact
    assert result == [a]
