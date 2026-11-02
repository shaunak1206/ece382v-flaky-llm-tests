import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == 2**63 - 1 or iv.end > 1_000_000  # maxsize is platform dependent int
    assert len(iv) == iv.end - iv.begin


def test_interval_creation_specific_values():
    iv = Interval(5, 10)
    assert iv.begin == 5
    assert iv.end == 10
    assert len(iv) == 5


def test_interval_len_zero_length():
    iv = Interval(7, 7)
    assert len(iv) == 0


def test_interval_comparison_lt_le_gt_ge():
    a = Interval(1, 5)
    b = Interval(3, 7)
    c = Interval(1, 2)
    assert (a < b) is True
    assert (a <= b) is True
    assert (b > a) is True
    assert (b >= a) is True
    # same begin equal comparison
    assert (a <= c) is True
    assert (a >= c) is True
    assert (a < c) is False
    assert (a > c) is False


def test_interval_contains_interval_inside():
    outer = Interval(2, 10)
    inner = Interval(3, 9)
    assert inner in outer
    # equal intervals
    assert outer in outer
    # interval partially outside
    left_out = Interval(0, 5)
    right_out = Interval(5, 11)
    assert left_out not in outer
    assert right_out not in outer


def test_interval_overlaps_method_and_function():
    a = Interval(1, 5)
    b = Interval(4, 7)
    c = Interval(5, 10)
    d = Interval(6, 8)

    assert a.overlaps(b) is True
    assert b.overlaps(a) is True
    assert a.overlaps(c) is False  # touching only, no overlap
    assert c.overlaps(d) is True
    # test interval_overlaps function matches method
    from intervaltree.interval import interval_overlaps
    assert interval_overlaps(a, b) == a.overlaps(b)
    assert interval_overlaps(c, d) == c.overlaps(d)
    assert interval_overlaps(a, c) == a.overlaps(c)


def test_interval_touches():
    a = Interval(1, 5)
    b = Interval(5, 10)
    c = Interval(0, 1)
    d = Interval(2, 3)
    e = Interval(6, 7)
    # touches if ends and begins meet exactly
    assert a.touches(b) is True
    assert c.touches(a) is True
    assert b.touches(a) is True
    # non-touching intervals
    assert a.touches(d) is False
    assert b.touches(e) is False


def test_combine_intervals_basic():
    a = Interval(1, 5)
    b = Interval(4, 10)
    combined = combine_intervals(a, b)
    assert isinstance(combined, Interval)
    assert combined.begin == 1
    assert combined.end == 10
    # commutative
    combined2 = combine_intervals(b, a)
    assert combined2.begin == 1
    assert combined2.end == 10


def test_combine_three_intervals_basic():
    a = Interval(1, 5)
    b = Interval(4, 10)
    c = Interval(0, 2)
    combined = combine_three_intervals(a, b, c)
    assert isinstance(combined, Interval)
    assert combined.begin == 0
    assert combined.end == 10
    # check order independence for values same set
    combined_alt = combine_three_intervals(c, a, b)
    assert combined_alt.begin == 0
    assert combined_alt.end == 10


def test_get_minimum_bounding_interval_basic():
    a = Interval(1, 10)
    b = Interval(5, 7)
    bbox = get_minimum_bounding_interval(a, b)
    assert isinstance(bbox, Interval)
    # minimum bounding contains both intervals totally
    assert bbox.begin <= a.begin and bbox.begin <= b.begin
    assert bbox.end >= a.end and bbox.end >= b.end
    # minimum interval that covers both begin and end
    assert bbox.begin == min(a.begin, b.begin)
    assert bbox.end == max(a.end, b.end)

    # reverse order same
    bbox_rev = get_minimum_bounding_interval(b, a)
    assert bbox == bbox_rev


def test_add_operator_combines_intervals():
    a = Interval(5, 15)
    b = Interval(10, 20)
    c = a + b
    assert isinstance(c, Interval)
    assert c.begin == 5
    assert c.end == 20

    d = b + a
    assert d == c


def test_get_overlap_function_behavior():
    from intervaltree.interval import get_overlap
    # b inside a
    a = Interval(2, 10)
    b = Interval(4, 8)
    overlap = get_overlap(a, b)
    assert overlap == b

    # b begins before a and overlaps partially on left
    b = Interval(0, 4)
    overlap = get_overlap(a, b)
    assert isinstance(overlap, Interval)
    assert overlap.begin == a.begin
    assert overlap.end == b.end
    # result must overlap both original intervals
    assert overlap.overlaps(a)
    assert overlap.overlaps(b)

    # b begins within a but ends after a - overlaps on right
    b = Interval(7, 12)
    overlap = get_overlap(a, b)
    assert isinstance(overlap, Interval)
    assert overlap.begin == b.begin
    assert overlap.end == a.end
    # overlap between a and b
    assert overlap.overlaps(a)
    assert overlap.overlaps(b)


def test_remove_method_no_overlap_returns_self():
    a = Interval(1, 10)
    b = Interval(20, 30)
    result = a.remove(b)
    # no overlaps, so result is list with original interval only
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == a


def test_remove_method_when_contained_returns_empty():
    a = Interval(5, 10)
    b = Interval(1, 20)
    # a in b: remove b from a means a covered, returns []
    result = a.remove(b)
    assert result == []


def test_remove_method_overlapping_splits():
    a = Interval(1, 10)
    # overlap partially in the middle
    b = Interval(4, 7)
    result = a.remove(b)
    # result should have two intervals: before the overlap and after the overlap
    assert isinstance(result, list)
    assert all(isinstance(iv, Interval) for iv in result)
    # intervals have nonzero length
    for iv in result:
        assert len(iv) > 0
    # The two intervals should cover all of a except the overlap part
    unioned = result[0] + result[1]
    assert unioned.begin == a.begin
    assert unioned.end == a.end
    # Check they do not overlap b
    for iv in result:
        assert not iv.overlaps(b)


def test_remove_method_overlapping_touching_interval():
    a = Interval(1, 5)
    # b touches a on right edge but does not overlap
    b = Interval(5, 8)
    result = a.remove(b)
    # touching does not count as overlapping, so original interval is returned
    assert result == [a]

    b2 = Interval(0, 1)
    result2 = a.remove(b2)
    assert result2 == [a]


def test_remove_method_exact_same_intervals():
    a = Interval(1, 5)
    b = Interval(1, 5)
    # b covers a fully, so removal returns []
    result = a.remove(b)
    assert result == []


def test_remove_method_overlap_at_beginning_only():
    a = Interval(10, 20)
    b = Interval(5, 12)
    result = a.remove(b)
    # Should remove the overlapping segment at beginning of "a"
    assert all(isinstance(iv, Interval) for iv in result)
    # The remaining intervals do not overlap with b
    for iv in result:
        assert not iv.overlaps(b)
    # Combined length of result intervals equals original length minus overlap
    sum_len = sum(len(iv) for iv in result)
    overlap_len = min(a.end, b.end) - max(a.begin, b.begin)
    assert pytest.approx(sum_len + overlap_len, rel=1e-12) == len(a)


def test_remove_method_overlap_at_end_only():
    a = Interval(10, 20)
    b = Interval(18, 25)
    result = a.remove(b)
    assert all(isinstance(iv, Interval) for iv in result)
    for iv in result:
        assert not iv.overlaps(b)
    sum_len = sum(len(iv) for iv in result)
    overlap_len = min(a.end, b.end) - max(a.begin, b.begin)
    assert pytest.approx(sum_len + overlap_len, rel=1e-12) == len(a)


def test_combine_intervals_with_equal_begin_and_end():
    a = Interval(5, 10)
    b = Interval(5, 10)
    combined = combine_intervals(a, b)
    assert combined.begin == 5
    assert combined.end == 10


def test_minimum_bounding_interval_with_equal_intervals():
    a = Interval(3, 7)
    b = Interval(3, 7)
    bbox = get_minimum_bounding_interval(a, b)
    assert bbox.begin == 3
    assert bbox.end == 7


def test_combine_three_intervals_with_nested_intervals():
    a = Interval(1, 10)
    b = Interval(2, 9)
    c = Interval(3, 8)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 10
