import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval

def test_interval_creation_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == 2**63 - 1 or iv.end > 0  # maxsize, platform dependent

def test_interval_creation_custom():
    iv = Interval(5, 10)
    assert iv.begin == 5
    assert iv.end == 10

def test_interval_length():
    iv = Interval(3, 8)
    assert len(iv) == 5

def test_interval_lt_and_le():
    a = Interval(1, 3)
    b = Interval(2, 4)
    assert a < b
    assert a <= b
    assert not (b < a)
    assert not (b <= a)
    c = Interval(1, 5)
    assert a <= c

def test_interval_gt_and_ge():
    a = Interval(5, 10)
    b = Interval(3, 8)
    assert a > b
    assert a >= b
    assert not (b > a)
    assert not (b >= a)
    c = Interval(5, 15)
    assert a >= c == False

def test_contains_interval_true():
    outer = Interval(2, 10)
    inner = Interval(3, 8)
    assert inner in outer
    assert not (outer in inner)

def test_contains_interval_false():
    outer = Interval(2, 10)
    not_contained = Interval(1, 5)
    assert not_contained not in outer

def test_overlaps_method_true():
    a = Interval(5, 10)
    b = Interval(8, 12)
    assert a.overlaps(b)
    assert b.overlaps(a)

def test_overlaps_method_false_touching():
    a = Interval(5, 10)
    b = Interval(10, 15)  # Touch but no overlap
    assert not a.overlaps(b)
    assert not b.overlaps(a)

def test_touches_true_and_false():
    a = Interval(5, 10)
    b = Interval(10, 13)
    c = Interval(11, 15)
    assert a.touches(b)
    assert not a.touches(c)
    assert b.touches(a)

def test_add_intervals():
    a = Interval(1, 5)
    b = Interval(3, 10)
    c = a + b
    assert c.begin == 1
    assert c.end == 10

def test_combine_intervals():
    a = Interval(1, 5)
    b = Interval(4, 9)
    c = combine_intervals(a, b)
    assert c.begin == 1
    assert c.end == 9

def test_combine_three_intervals():
    a = Interval(1,3)
    b = Interval(2,6)
    c = Interval(4,10)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 10

def test_get_minimum_bounding_interval():
    a = Interval(5, 7)
    b = Interval(3, 8)
    bounding = get_minimum_bounding_interval(a, b)
    assert bounding.begin == 3
    assert bounding.end == 8

def test_remove_no_overlap_returns_self():
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]

def test_remove_fully_contained_returns_empty():
    a = Interval(3, 8)
    b = Interval(1, 10)
    result = a.remove(b)
    assert result == []

def test_remove_partial_left_overlap():
    a = Interval(3, 10)
    b = Interval(1, 5)
    result = a.remove(b)
    assert len(result) == 1
    assert result[0] == Interval(5, 10)

def test_remove_partial_right_overlap():
    a = Interval(3, 10)
    b = Interval(8, 12)
    result = a.remove(b)
    assert len(result) == 1
    assert result[0] == Interval(3, 8)

def test_remove_exact_overlap():
    a = Interval(3, 10)
    b = Interval(3, 10)
    result = a.remove(b)
    assert result == []

def test_remove_overlapping_two_segments_filtered():
    # Manually check that empty segments are not returned
    a = Interval(1, 10)
    b = Interval(4, 6)
    result = a.remove(b)
    assert len(result) == 2
    assert Interval(1,4) in result
    assert Interval(6,10) in result

def test_get_overlap_contained():
    a = Interval(2, 10)
    b = Interval(4, 8)
    overlap = a._get_overlap(b)
    # b is fully inside a, overlap returns b
    assert overlap == b

def test_get_overlap_left_side():
    a = Interval(4, 10)
    b = Interval(1, 6)
    overlap = a._get_overlap(b)
    assert overlap == Interval(4, 6)

def test_get_overlap_right_side():
    a = Interval(1, 6)
    b = Interval(4, 10)
    overlap = a._get_overlap(b)
    assert overlap == Interval(4, 6)

def test_interval_order_equal_begin_diff_end():
    a = Interval(1,5)
    b = Interval(1,7)
    assert a <= b
    assert not a < b
    assert b > a

def test_sorted_points_not_used_in_functions():
    # Confirm that sorted(points) returns a sorted list but isn't assigned back (bug in original)
    from intervaltree import interval
    a = Interval(5,7)
    b = Interval(1,10)
    # This test just ensures function doesn't error and result is correct even if sorted() not assigned
    res = interval.get_minimum_bounding_interval(a,b)
    assert res.begin == 1
    assert res.end == 10
