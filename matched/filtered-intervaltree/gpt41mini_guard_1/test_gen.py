import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    interval = Interval()
    assert interval.begin == 0
    # maxsize from sys module is platform dependent, test type and range at least
    assert isinstance(interval.end, int)
    assert interval.end > interval.begin


def test_interval_creation_with_values():
    interval = Interval(5, 10)
    assert interval.begin == 5
    assert interval.end == 10


def test_interval_lt_and_le_and_gt_and_ge():
    a = Interval(1, 5)
    b = Interval(3, 7)

    assert (a < b) is True
    assert (a <= b) is True
    assert (b > a) is True
    assert (b >= a) is True

    c = Interval(1, 4)
    # begin equal
    assert (a <= c) is True
    assert (a >= c) is True
    assert not (a < c)
    assert not (a > c)


def test_interval_len():
    a = Interval(10, 20)
    assert len(a) == 10
    b = Interval(0, 0)
    assert len(b) == 0
    c = Interval(-5, 5)
    assert len(c) == 10


def test_interval_contains_interval():
    a = Interval(1, 10)
    b = Interval(3, 8)
    c = Interval(0, 5)
    d = Interval(5, 15)

    assert b in a
    assert not c in a  # begin before a.begin
    assert not d in a  # end after a.end


def test_interval_addition_combine():
    a = Interval(1, 5)
    b = Interval(4, 10)
    c = a + b
    # Combined interval covers both inputs
    assert c.begin == 1
    assert c.end == 10
    d = b + a
    assert d == c  # commutative


def test_combine_intervals_function():
    a = Interval(2, 6)
    b = Interval(4, 10)
    combined = combine_intervals(a, b)
    assert combined.begin == 2
    assert combined.end == 10

    # reversed order same result
    combined_rev = combine_intervals(b, a)
    assert combined_rev == combined


def test_combine_three_intervals_function():
    a = Interval(1, 3)
    b = Interval(5, 7)
    c = Interval(2, 6)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 7

    # different order same result
    combined2 = combine_three_intervals(c, a, b)
    assert combined2 == combined


def test_get_minimum_bounding_interval_function():
    a = Interval(0, 5)
    b = Interval(3, 10)
    bounding = get_minimum_bounding_interval(a, b)
    # Should be min begin and max end among two intervals
    assert bounding.begin == 0
    assert bounding.end == 10

    # reversed order same result
    bounding2 = get_minimum_bounding_interval(b, a)
    assert bounding2 == bounding


def test_overlaps_method_true_and_false():
    a = Interval(1, 5)
    b = Interval(4, 8)
    c = Interval(5, 10)
    d = Interval(0, 1)

    assert a.overlaps(b) is True  # overlapping
    assert b.overlaps(a) is True  # symmetric
    assert not a.overlaps(c)      # they touch but do not overlap (end == begin)
    assert not a.overlaps(d)      # touching at 1==1, not overlapping


def test_touches_method_true_and_false():
    a = Interval(1, 5)
    b = Interval(5, 10)
    c = Interval(0, 1)
    d = Interval(3, 4)

    assert a.touches(b) is True   # end == begin
    assert a.touches(c) is True   # begin == end
    assert not a.touches(d)       # overlap, but no touching at boundaries
    assert not b.touches(d)       # no touching


def test_get_overlap_on_contained_interval():
    a = Interval(1, 10)
    b = Interval(3, 7)
    overlap = a._get_overlap(b)
    assert overlap == b
    # Overlap length correct and interval contained in a
    assert overlap.begin >= a.begin
    assert overlap.end <= a.end


def test_get_overlap_other_cases():
    a = Interval(5, 15)
    b = Interval(0, 7)  # begins before a
    overlap = a._get_overlap(b)
    assert overlap.begin == a.begin
    assert overlap.end == b.end

    b2 = Interval(12, 20)  # begins after a
    overlap = a._get_overlap(b2)
    assert overlap.begin == b2.begin
    assert overlap.end == a.end


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (Interval(1, 10), Interval(3, 7), True),
        (Interval(1, 5), Interval(5, 10), False),
        (Interval(0, 0), Interval(0, 0), False),
        (Interval(2, 6), Interval(6, 8), False),
    ]
)
def test_interval_overlaps_function_matches_method(a, b, expected):
    # Test that interval_overlaps function matches Interval.overlaps method
    assert a.overlaps(b) == expected
    from intervaltree.interval import interval_overlaps
    assert interval_overlaps(a, b) == expected


def test_remove_no_overlap_returns_self():
    a = Interval(10, 20)
    b = Interval(30, 40)
    result = a.remove(b)
    assert result == [a]  # original interval unchanged


def test_remove_full_containment_returns_empty():
    a = Interval(10, 20)
    b = Interval(5, 25)
    result = a.remove(b)
    # a contained in b -> empty list removed entirely
    assert result == []


def test_remove_partial_overlap_returns_two_segments():
    a = Interval(10, 20)
    b = Interval(15, 25)
    result = a.remove(b)
    assert len(result) == 1 or len(result) == 2
    # Check that all returned intervals do not overlap b
    for seg in result:
        assert not seg.overlaps(b)
    # All returned intervals are within a's boundaries
    for seg in result:
        assert seg.begin >= a.begin
        assert seg.end <= a.end


def test_remove_partial_overlap_internal_structure():
    # Specifically test removal where overlap cuts middle and returns two intervals
    a = Interval(10, 30)
    b = Interval(15, 25)
    result = a.remove(b)
    # Both segments are non-empty and disjoint from b
    assert all(isinstance(seg, Interval) for seg in result)
    assert all(len(seg) > 0 for seg in result)
    # The segments cover parts of a outside b
    begins = [seg.begin for seg in result]
    ends = [seg.end for seg in result]
    assert min(begins) >= a.begin
    assert max(ends) <= a.end
    # Neither segment overlaps b
    for seg in result:
        assert not seg.overlaps(b)


def test_remove_partial_overlap_left_only():
    # Interval to remove overlaps only the left part
    a = Interval(10, 20)
    b = Interval(5, 15)
    result = a.remove(b)
    # Only the right segment [15,20) remains
    assert len(result) == 1
    seg = result[0]
    assert seg.begin == 15
    assert seg.end == 20


def test_remove_partial_overlap_right_only():
    a = Interval(10, 20)
    b = Interval(15, 25)
    result = a.remove(b)
    # Only the left segment [10, 15) remains
    assert len(result) == 1
    seg = result[0]
    assert seg.begin == 10
    assert seg.end == 15


def test_remove_touching_but_not_overlapping():
    a = Interval(10, 20)
    b = Interval(20, 30)
    result = a.remove(b)
    # They touch but do not overlap, the original interval should remain
    assert result == [a]
