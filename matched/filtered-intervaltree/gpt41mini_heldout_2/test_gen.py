import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == 2**63 - 1 or iv.end > 0  # maxsize varies by platform


def test_interval_creation_explicit():
    iv = Interval(5, 10)
    assert iv.begin == 5
    assert iv.end == 10


def test_interval_length():
    iv = Interval(4, 9)
    assert len(iv) == 5
    iv2 = Interval(0, 0)
    assert len(iv2) == 0


def test_interval_comparisons():
    iv1 = Interval(1, 3)
    iv2 = Interval(2, 5)
    assert (iv1 < iv2) is True
    assert (iv1 <= iv2) is True
    assert (iv2 > iv1) is True
    assert (iv2 >= iv1) is True
    assert (iv1 <= iv1) is True
    assert (iv1 >= iv1) is True


def test_interval_contains():
    outer = Interval(2, 10)
    inner = Interval(3, 9)
    not_inner = Interval(1, 9)
    assert inner in outer
    assert not (not_inner in outer)


def test_interval_overlaps_method_true():
    iv1 = Interval(1, 5)
    iv2 = Interval(4, 8)
    iv3 = Interval(5, 10)
    # iv1 and iv2 overlap
    assert iv1.overlaps(iv2) is True
    # iv1 and iv3 just touch at 5, overlaps should be False
    assert iv1.overlaps(iv3) is False


def test_interval_overlaps_function_consistency():
    iv1 = Interval(1, 3)
    iv2 = Interval(2, 4)
    assert iv1.overlaps(iv2) == iv2.overlaps(iv1)
    # Should match symmetry and commutativity


def test_touches_method_true_and_false():
    iv1 = Interval(3, 7)
    iv2 = Interval(7, 10)
    iv3 = Interval(0, 3)
    iv4 = Interval(5, 6)
    assert iv1.touches(iv2) is True
    assert iv3.touches(iv1) is True
    assert iv1.touches(iv4) is False


def test_add_operator_combine():
    iv1 = Interval(1, 4)
    iv2 = Interval(3, 7)
    combined = iv1 + iv2
    assert combined.begin == 1
    assert combined.end == 7


def test_combine_intervals_function():
    iv1 = Interval(2, 8)
    iv2 = Interval(5, 10)
    combined = combine_intervals(iv1, iv2)
    assert combined.begin == 2
    assert combined.end == 10


def test_combine_three_intervals_function():
    iv1 = Interval(1, 3)
    iv2 = Interval(5, 7)
    iv3 = Interval(2, 6)
    combined = combine_three_intervals(iv1, iv2, iv3)
    assert combined.begin == 1
    assert combined.end == 7


def test_get_minimum_bounding_interval():
    iv1 = Interval(4, 9)
    iv2 = Interval(6, 15)
    bounding = get_minimum_bounding_interval(iv1, iv2)
    # Should be from min(begin) to max(end)
    assert bounding.begin == 4
    assert bounding.end == 15


def test_remove_non_overlapping_returns_self():
    iv1 = Interval(1, 5)
    iv2 = Interval(6, 8)
    result = iv1.remove(iv2)
    assert result == [iv1]


def test_remove_contained_interval_returns_empty():
    iv1 = Interval(3, 10)
    iv2 = Interval(1, 12)
    # iv1 is contained in iv2, so iv1 in iv2 is True -> should return []
    result = iv1.remove(iv2)
    assert result == []


def test_remove_overlapping_returns_segments():
    iv1 = Interval(3, 10)
    iv2 = Interval(5, 7)
    # iv2 does not contain iv1, but they overlap
    result = iv1.remove(iv2)
    # Should produce two intervals: [3,5) and [7,10)
    assert Interval(3, 5) in result
    assert Interval(7, 10) in result
    # Lengths must be > 0
    for r in result:
        assert len(r) > 0
    assert len(result) == 2


def test_remove_overlap_touching():
    iv1 = Interval(3, 7)
    iv2 = Interval(7, 10)
    result = iv1.remove(iv2)
    # No overlap, touching at boundary, should return original interval
    assert result == [iv1]


def test_get_overlap_when_contained_returns_inner_interval():
    iv1 = Interval(2, 8)
    iv2 = Interval(3, 5)
    overlap = iv1._get_overlap(iv2)
    assert overlap == iv2


def test_get_overlap_b_begins_before_a():
    iv1 = Interval(5, 9)
    iv2 = Interval(3, 7)
    overlap = iv1._get_overlap(iv2)
    assert overlap.begin == 5
    assert overlap.end == 7


def test_get_overlap_b_begins_after_a():
    iv1 = Interval(3, 8)
    iv2 = Interval(5, 10)
    overlap = iv1._get_overlap(iv2)
    assert overlap.begin == 5
    assert overlap.end == 8


def test_multiple_combine_intervals_commutativity():
    iv1 = Interval(1, 5)
    iv2 = Interval(3, 7)
    combined1 = combine_intervals(iv1, iv2)
    combined2 = combine_intervals(iv2, iv1)
    assert combined1 == combined2

    combined3 = iv1 + iv2
    combined4 = iv2 + iv1
    assert combined3 == combined4


def test_interval_contains_itself():
    iv = Interval(4, 10)
    assert iv in iv

