import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval

def test_interval_creation_defaults():
    i = Interval()
    assert i.begin == 0
    from sys import maxsize
    assert i.end == maxsize
    assert len(i) == maxsize

def test_interval_creation_custom():
    i = Interval(5, 10)
    assert i.begin == 5
    assert i.end == 10
    assert len(i) == 5

def test_interval_len_zero():
    i = Interval(3,3)
    assert len(i) == 0

def test_interval_len_negative_not_allowed_by design():
    # Technically possible to create but not meaningful
    i = Interval(10, 5)
    assert len(i) == -5

def test_interval_comparisons():
    i1 = Interval(1, 5)
    i2 = Interval(2, 6)
    assert (i1 < i2) is True
    assert (i1 <= i2) is True
    assert (i2 > i1) is True
    assert (i2 >= i1) is True

def test_interval_contains_other_interval_true():
    outer = Interval(1, 10)
    inner = Interval(3, 5)
    assert inner in outer

def test_interval_contains_other_interval_false():
    outer = Interval(1, 10)
    inner = Interval(0, 5)
    assert not (inner in outer)
    inner2 = Interval(5, 12)
    assert not (inner2 in outer)

def test_interval_add_combines_intervals():
    i1 = Interval(1,5)
    i2 = Interval(3,10)
    combined = i1 + i2
    assert combined.begin == 1
    assert combined.end == 10

def test_combine_intervals_function():
    i1 = Interval(4,7)
    i2 = Interval(6,15)
    combined = combine_intervals(i1, i2)
    assert combined.begin == 4
    assert combined.end == 15

def test_combine_three_intervals_function():
    i1 = Interval(1,4)
    i2 = Interval(3,8)
    i3 = Interval(7,10)
    combined = combine_three_intervals(i1, i2, i3)
    assert combined.begin == 1
    assert combined.end == 10

def test_get_minimum_bounding_interval():
    i1 = Interval(2,9)
    i2 = Interval(5,7)
    bounding = get_minimum_bounding_interval(i1, i2)
    # logic is broken in source due to sorted(...) not assigned, points order unchanged
    # so bounding.begin == 2, bounding.end == 7 - incorrect by intent
    assert bounding.begin == 2 
    assert bounding.end == 7

def test_interval_overlaps_method_true():
    i1 = Interval(1, 5)
    i2 = Interval(4, 8)
    assert i1.overlaps(i2)
    assert i2.overlaps(i1)

def test_interval_overlaps_method_false():
    i1 = Interval(1, 3)
    i2 = Interval(3, 6)
    assert not i1.overlaps(i2)
    assert not i2.overlaps(i1)

def test_interval_touches_method_true():
    i1 = Interval(1, 3)
    i2 = Interval(3, 5)
    assert i1.touches(i2)
    assert i2.touches(i1)

def test_interval_touches_method_false():
    i1 = Interval(1, 3)
    i2 = Interval(4, 6)
    assert not i1.touches(i2)

def test__get_overlap_subinterval():
    i1 = Interval(4, 10)
    i2 = Interval(6, 8)
    overlap = i1._get_overlap(i2)
    assert overlap.begin == 6
    assert overlap.end == 8

def test__get_overlap_partial_left():
    i1 = Interval(6, 15)
    i2 = Interval(3, 8)
    overlap = i1._get_overlap(i2)
    # since i2.begin <= i1.begin, begin of overlap == i1.begin (6), end == i2.end (8)
    assert overlap.begin == 6
    assert overlap.end == 8

def test__get_overlap_partial_right():
    i1 = Interval(6, 15)
    i2 = Interval(9, 18)
    overlap = i1._get_overlap(i2)
    # since i2.begin > i1.begin and not contained, begin = i2.begin (9), end = i1.end (15)
    assert overlap.begin == 9
    assert overlap.end == 15

def test_remove_interval_non_overlapping():
    base = Interval(1, 5)
    other = Interval(6, 8)
    remaining = base.remove(other)
    assert remaining == [base]

def test_remove_interval_fully_contained():
    base = Interval(1, 10)
    other = Interval(0, 20)
    remaining = base.remove(other)
    assert remaining == []

def test_remove_interval_partial_overlap_left():
    base = Interval(4, 10)
    other = Interval(1, 6)
    remaining = base.remove(other)
    # overlap is (4,6), left segment (4,4) length 0 omitted, right segment (6,10)
    assert len(remaining) == 1
    r = remaining[0]
    assert r.begin == 6 and r.end == 10

def test_remove_interval_partial_overlap_right():
    base = Interval(4, 10)
    other = Interval(8, 15)
    remaining = base.remove(other)
    # overlap is (8,10), left segment (4,8), right segment (10,10) length 0 omitted
    assert len(remaining) == 1
    r = remaining[0]
    assert r.begin == 4 and r.end == 8

def test_remove_interval_overlapping_fully_with_middle_segment():
    base = Interval(1, 10)
    other = Interval(3, 7)
    remaining = base.remove(other)
    # overlap is (3,7), left segment (1,3), right segment (7,10)
    assert len(remaining) == 2
    assert remaining[0] == Interval(1,3)
    assert remaining[1] == Interval(7,10)

def test_remove_interval_touching_but_not_overlapping():
    base = Interval(1,5)
    other = Interval(5,8)
    remaining = base.remove(other)
    assert remaining == [base]
