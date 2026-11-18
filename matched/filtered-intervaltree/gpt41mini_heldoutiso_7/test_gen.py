import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_defaults():
    interval = Interval()
    assert interval.begin == 0
    assert interval.end == 2**63 - 1 or interval.end > 0  # maxsize depending on platform


def test_interval_creation_custom():
    interval = Interval(5, 10)
    assert interval.begin == 5
    assert interval.end == 10


def test_interval_ordering_lt():
    a = Interval(1, 5)
    b = Interval(2, 6)
    assert (a < b) is True
    assert (b < a) is False


def test_interval_ordering_le():
    a = Interval(1, 5)
    b = Interval(1, 6)
    c = Interval(0, 3)
    assert (a <= b) is True
    assert (b <= a) is False
    assert (c <= a) is True


def test_interval_ordering_gt_ge():
    a = Interval(5, 10)
    b = Interval(3, 4)
    assert (a > b) is True
    assert (a >= b) is True
    assert (b >= a) is False


def test_interval_contains_interval():
    outer = Interval(1, 10)
    inner = Interval(3, 7)
    not_contained = Interval(0, 5)
    assert inner in outer
    assert not (not_contained in outer)
    # edge case: exact match
    exact = Interval(1, 10)
    assert exact in outer


def test_interval_len():
    interval = Interval(10, 20)
    assert len(interval) == 10
    interval_zero_len = Interval(5, 5)
    assert len(interval_zero_len) == 0
    interval_negative_len = Interval(10, 5)
    assert len(interval_negative_len) == -5


def test_interval_add_combines_correctly():
    a = Interval(1, 5)
    b = Interval(3, 10)
    combined = a + b
    assert combined.begin == 1
    assert combined.end == 10

    a = Interval(6, 8)
    b = Interval(1, 2)
    combined = a + b
    assert combined.begin == 1
    assert combined.end == 8


def test_combine_intervals_function():
    a = Interval(1, 4)
    b = Interval(3, 7)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 7

    a = Interval(5, 10)
    b = Interval(1, 2)
    combined = combine_intervals(a, b)
    assert combined.begin == 1
    assert combined.end == 10


def test_combine_three_intervals_function():
    a = Interval(1, 2)
    b = Interval(4, 5)
    c = Interval(3, 6)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 1
    assert combined.end == 6

    a = Interval(10, 20)
    b = Interval(15, 25)
    c = Interval(5, 8)
    combined = combine_three_intervals(a, b, c)
    assert combined.begin == 5
    assert combined.end == 25


def test_get_minimum_bounding_interval_function():
    a = Interval(2, 5)
    b = Interval(4, 10)
    result = get_minimum_bounding_interval(a, b)
    # Because in the source code, "sorted(points)" is called but its result is not assigned,
    # points is not sorted and points[0] is interval_a.begin, points[-1] is interval_b.end
    assert result.begin == 2
    assert result.end == 10

    a = Interval(10, 15)
    b = Interval(5, 12)
    result = get_minimum_bounding_interval(a, b)
    assert result.begin == 10
    assert result.end == 12


def test_interval_overlaps_method_and_function():
    a = Interval(1, 5)
    b = Interval(4, 8)
    c = Interval(5, 10)
    d = Interval(10, 15)

    # Overlap
    assert a.overlaps(b) is True
    assert b.overlaps(a) is True

    # Touching but not overlapping
    assert a.overlaps(c) is False
    assert c.overlaps(a) is False

    # No overlap
    assert a.overlaps(d) is False
    assert d.overlaps(a) is False


def test_interval_touches():
    a = Interval(1, 5)
    b = Interval(5, 10)
    c = Interval(0, 1)
    d = Interval(6, 7)

    assert a.touches(b) is True
    assert b.touches(a) is True
    assert a.touches(c) is True
    assert c.touches(a) is True
    assert a.touches(d) is False


def test_get_overlap_private_method():
    a = Interval(1, 6)
    b = Interval(3, 5)
    overlap = a._get_overlap(b)
    assert overlap.begin == 3
    assert overlap.end == 5

    a = Interval(3, 5)
    b = Interval(1, 6)
    overlap = a._get_overlap(b)
    assert overlap.begin == 3
    assert overlap.end == 6

    a = Interval(1, 6)
    b = Interval(6, 8)
    overlap = a._get_overlap(b)
    # No overlap: b.begin == a.end
    # Method logic returns Interval(begin, end) with values based on function, no explicit check for no overlap
    assert isinstance(overlap, Interval)


@pytest.mark.parametrize(
    "self_interval, other_interval, expected_result",
    [
        (Interval(1, 10), Interval(2, 5), []),               # other contains self: False
        (Interval(2, 5), Interval(1, 10), []),               # self contained in other -> return []
        (Interval(1, 10), Interval(5, 15), [Interval(1, 5)]),# overlaps with other
        (Interval(1, 10), Interval(11, 15), [Interval(1, 10)]),  # no overlap, return self
        (Interval(1, 5), Interval(3, 5), [Interval(1, 3)]),  # overlap on right edge.
        (Interval(1, 5), Interval(0, 2), [Interval(2, 5)]),  # overlap on left edge
        (Interval(1, 10), Interval(1, 10), []),              # identical intervals
    ]
)
def test_interval_remove(self_interval, other_interval, expected_result):
    result = self_interval.remove(other_interval)
    assert result == expected_result


def test_remove_non_overlapping_returns_self():
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]


def test_remove_full_containment_returns_empty():
    a = Interval(5, 10)
    b = Interval(1, 20)
    result = a.remove(b)
    assert result == []


def test_remove_partial_overlap_left():
    a = Interval(5, 15)
    b = Interval(1, 10)
    result = a.remove(b)
    assert len(result) == 1
    # The remaining segment should be after the overlap
    assert result[0].begin == 10
    assert result[0].end == 15


def test_remove_partial_overlap_right():
    a = Interval(5, 15)
    b = Interval(10, 20)
    result = a.remove(b)
    assert len(result) == 1
    # The remaining segment should be before the overlap
    assert result[0].begin == 5
    assert result[0].end == 10


def test_interval_contains_itself():
    a = Interval(1, 5)
    assert a in a


def test_interval_len_calculation():
    a = Interval(10, 20)
    assert len(a) == 10
    b = Interval(5, 5)
    assert len(b) == 0
    c = Interval(20, 10)
    assert len(c) == -10


def test_add_overlapping_and_disjoint_intervals():
    a = Interval(1, 5)
    b = Interval(4, 10)
    c = Interval(10, 15)

    added = a + b
    assert added == Interval(1, 10)

    added_disjoint = a + c
    assert added_disjoint == Interval(1, 15)


def test_touching_intervals():
    a = Interval(0, 5)
    b = Interval(5, 10)
    c = Interval(10, 15)
    d = Interval(6, 9)

    assert a.touches(b) is True
    assert b.touches(a) is True
    assert b.touches(c) is True
    assert c.touches(b) is True
    assert a.touches(c) is False
    assert b.touches(d) is False


def test_overlaps_not_interval_instance():
    # Overlaps expects Interval arguments, but no explicit type checks
    a = Interval(1, 5)
    with pytest.raises(AttributeError):
        a.overlaps(None)  # None has no begin attribute


def test_contains_not_interval_instance():
    a = Interval(1, 5)
    with pytest.raises(AttributeError):
        _ = None in a

