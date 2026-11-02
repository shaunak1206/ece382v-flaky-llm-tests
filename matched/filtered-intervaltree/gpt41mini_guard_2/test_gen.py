import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    iv = Interval()
    assert iv.begin == 0
    # maxsize varies by platform but is fixed; just check type and ordering
    assert iv.end > iv.begin


def test_interval_creation_values():
    iv = Interval(5, 10)
    assert iv.begin == 5
    assert iv.end == 10


def test_interval_length_positive():
    iv = Interval(3, 8)
    assert len(iv) == 5


def test_interval_length_zero():
    iv = Interval(7, 7)
    assert len(iv) == 0


def test_interval_comparison_lt_and_le():
    iv1 = Interval(1, 5)
    iv2 = Interval(3, 8)
    assert iv1 < iv2
    assert iv1 <= iv2
    assert not (iv2 < iv1)
    assert not (iv2 <= iv1)
    # equals on begin counts in le
    iv3 = Interval(1, 7)
    assert iv1 <= iv3
    assert not iv1 < iv3


def test_interval_comparison_gt_and_ge():
    iv1 = Interval(10, 15)
    iv2 = Interval(5, 9)
    assert iv1 > iv2
    assert iv1 >= iv2
    assert not (iv2 > iv1)
    assert not (iv2 >= iv1)
    iv3 = Interval(10, 12)
    assert iv1 >= iv3
    assert not iv1 > iv3


def test_interval_contains_true_and_false():
    container = Interval(2, 10)
    inside = Interval(3, 9)
    outside_begin = Interval(1, 5)
    outside_end = Interval(5, 11)

    assert inside in container
    assert not (outside_begin in container)
    assert not (outside_end in container)
    # identical intervals
    same = Interval(2, 10)
    assert same in container


def test_interval_overlaps_method_true_and_false():
    iv1 = Interval(1, 5)
    iv2 = Interval(4, 7)
    iv3 = Interval(5, 10)
    iv4 = Interval(10, 15)

    assert iv1.overlaps(iv2)
    assert iv2.overlaps(iv1)
    assert not iv1.overlaps(iv3)  # 5 == 5 but touching not overlap
    assert not iv3.overlaps(iv4)
    assert not iv1.overlaps(iv4)


def test_interval_touches_true_and_false():
    iv1 = Interval(1, 5)
    iv2 = Interval(5, 10)
    iv3 = Interval(0, 1)
    iv4 = Interval(2, 4)

    assert iv1.touches(iv2)
    assert iv2.touches(iv1)
    assert iv3.touches(iv1)
    assert not iv1.touches(iv4)
    assert not iv4.touches(iv1)


def test_interval_add_merges_correctly():
    iv1 = Interval(3, 8)
    iv2 = Interval(6, 10)
    result = iv1 + iv2
    assert result.begin == 3
    assert result.end == 10
    # addition is commutative
    result2 = iv2 + iv1
    assert result == result2


def test_combine_intervals_basic():
    iv1 = Interval(2, 6)
    iv2 = Interval(4, 8)
    combined = combine_intervals(iv1, iv2)
    assert combined.begin == 2
    assert combined.end == 8


def test_combine_intervals_disjoint():
    iv1 = Interval(1, 3)
    iv2 = Interval(10, 15)
    combined = combine_intervals(iv1, iv2)
    assert combined.begin == 1
    assert combined.end == 15


def test_combine_three_intervals_order_irrelevant():
    iv1 = Interval(1, 5)
    iv2 = Interval(4, 10)
    iv3 = Interval(2, 7)
    combined = combine_three_intervals(iv1, iv2, iv3)
    assert combined.begin == 1
    assert combined.end == 10

    combined_alt = combine_three_intervals(iv3, iv1, iv2)
    assert combined_alt == combined


def test_get_minimum_bounding_interval_basic():
    iv1 = Interval(5, 9)
    iv2 = Interval(3, 12)
    bounding = get_minimum_bounding_interval(iv1, iv2)
    assert bounding.begin == 3
    assert bounding.end == 12

    bounding2 = get_minimum_bounding_interval(iv2, iv1)
    assert bounding2 == bounding


def test_interval_remove_non_overlapping_returns_self():
    iv1 = Interval(10, 20)
    iv2 = Interval(21, 25)
    result = iv1.remove(iv2)
    assert len(result) == 1
    assert result[0] == iv1


def test_interval_remove_fully_contained_returns_empty():
    iv1 = Interval(10, 20)
    iv2 = Interval(5, 25)
    # iv1 is fully contained in iv2, so iv1 in iv2 True, so remove returns []
    assert iv1.remove(iv2) == []


def test_interval_remove_partial_overlap_two_segments():
    iv1 = Interval(10, 20)
    iv2 = Interval(15, 25)  # overlapping on right side
    result = iv1.remove(iv2)
    assert len(result) == 1
    segment = result[0]
    # segment should be the left part before overlap
    assert segment.begin == 10
    assert segment.end == 15

    iv3 = Interval(5, 12)  # overlapping on left side
    result2 = iv1.remove(iv3)
    assert len(result2) == 1
    segment2 = result2[0]
    assert segment2.begin == 12
    assert segment2.end == 20


def test_interval_remove_overlap_split_cuts_both_sides():
    iv1 = Interval(10, 30)
    iv2 = Interval(15, 20)
    result = iv1.remove(iv2)
    # removal of strictly contained interval yields two pieces
    # But our code returns two segments with filtering zero length ones
    # (overlap is 15-20)
    assert len(result) == 2
    left, right = sorted(result, key=lambda x: x.begin)
    assert left.begin == 10
    assert left.end == 15
    assert right.begin == 20
    assert right.end == 30


def test_interval_remove_touching_returns_self():
    iv1 = Interval(10, 15)
    iv2 = Interval(15, 20)
    # Touching is not overlapping, so remove returns [iv1]
    result = iv1.remove(iv2)
    assert result == [iv1]

    iv3 = Interval(5, 10)
    result2 = iv1.remove(iv3)
    assert result2 == [iv1]


def test_get_minimum_bounding_interval_does_not_modify_originals():
    iv1 = Interval(1, 2)
    iv2 = Interval(3, 4)
    bounding = get_minimum_bounding_interval(iv1, iv2)
    assert bounding != iv1
    assert bounding != iv2
    assert iv1.begin == 1 and iv1.end == 2
    assert iv2.begin == 3 and iv2.end == 4


def test_combine_intervals_with_equal_begins_and_ends():
    iv1 = Interval(5, 10)
    iv2 = Interval(5, 10)
    combined = combine_intervals(iv1, iv2)
    assert combined.begin == 5
    assert combined.end == 10


def test_combine_three_intervals_with_equal_points():
    iv1 = Interval(1, 5)
    iv2 = Interval(1, 5)
    iv3 = Interval(1, 5)
    combined = combine_three_intervals(iv1, iv2, iv3)
    assert combined.begin == 1
    assert combined.end == 5
