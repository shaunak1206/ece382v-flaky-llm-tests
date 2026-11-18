import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_default():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == 2**63 - 1 or iv.end > 1_000_000  # maxsize typically large


def test_interval_creation_custom():
    iv = Interval(1, 5)
    assert iv.begin == 1
    assert iv.end == 5


def test_interval_length():
    iv = Interval(3, 7)
    assert len(iv) == 4


def test_interval_comparison_lt_le_gt_ge():
    a = Interval(1, 5)
    b = Interval(2, 6)
    assert a < b
    assert a <= b
    assert not a > b
    assert not a >= b
    assert b > a
    assert b >= a
    assert not b < a
    assert not b <= a


def test_interval_contains_operator_true_false():
    a = Interval(1, 10)
    b = Interval(3, 5)
    c = Interval(0, 4)
    d = Interval(5, 15)

    assert b in a
    assert not a in b
    assert not c in a
    assert not d in a


def test_interval_add_operator():
    a = Interval(1, 4)
    b = Interval(3, 6)
    c = a + b
    assert isinstance(c, Interval)
    assert c.begin == 1
    assert c.end == 6


def test_combine_intervals():
    a = Interval(5, 10)
    b = Interval(8, 15)
    c = combine_intervals(a, b)
    assert isinstance(c, Interval)
    assert c.begin == 5
    assert c.end == 15

    d = Interval(12, 20)
    e = combine_intervals(b, d)
    assert e.begin == 8
    assert e.end == 20


def test_combine_three_intervals():
    a = Interval(1, 4)
    b = Interval(3, 7)
    c = Interval(6, 10)
    combined = combine_three_intervals(a, b, c)
    assert isinstance(combined, Interval)
    assert combined.begin == 1
    assert combined.end == 10


def test_get_minimum_bounding_interval():
    a = Interval(2, 5)
    b = Interval(4, 9)
    bounding = get_minimum_bounding_interval(a, b)
    assert isinstance(bounding, Interval)
    # The code has a bug: sorted(points) does not assign, so points stays unsorted.
    # We test whether bounding.begin == smallest and bounding.end == largest among points.
    expected_begin = min(a.begin, a.end, b.begin, b.end)
    expected_end = max(a.begin, a.end, b.begin, b.end)
    assert bounding.begin == expected_begin
    assert bounding.end == expected_end


@pytest.mark.parametrize("a,b,expected", [
    (Interval(1, 5), Interval(3, 7), True),
    (Interval(1, 3), Interval(3, 5), False),
    (Interval(4, 6), Interval(1, 4), False),
    (Interval(2, 8), Interval(0, 3), True),
])
def test_interval_overlaps_method(a, b, expected):
    assert a.overlaps(b) == expected


def test_interval_touches():
    a = Interval(1, 5)
    b = Interval(5, 10)
    c = Interval(6, 12)
    d = Interval(0, 1)
    e = Interval(10, 20)

    assert a.touches(b)
    assert not a.touches(c)
    assert d.touches(a)
    assert not b.touches(c)
    assert not c.touches(e)


def test_remove_no_overlap_returns_self():
    a = Interval(1, 5)
    b = Interval(6, 10)
    result = a.remove(b)
    assert result == [a]


def test_remove_other_contains_self_returns_empty():
    a = Interval(3, 7)
    b = Interval(1, 10)
    result = a.remove(b)
    assert result == []


def test_remove_partial_overlap():
    a = Interval(1, 10)
    b = Interval(5, 15)
    result = a.remove(b)
    # Overlap segment is from 5 to 10, so remaining segment is from 1 to 5
    assert len(result) == 1
    assert all(isinstance(x, Interval) for x in result)
    assert result[0].begin == 1
    assert result[0].end == 5


def test_remove_partial_overlap_start():
    a = Interval(5, 15)
    b = Interval(1, 10)
    result = a.remove(b)
    # Overlap segment from 5 to 10, remaining from 10 to 15
    assert len(result) == 1
    assert result[0].begin == 10
    assert result[0].end == 15


def test_remove_subinterval_removes_both_segments():
    a = Interval(1, 10)
    b = Interval(3, 7)
    result = a.remove(b)
    # Remove [3,7] from [1,10] leaves [1,3) and (7,10]
    assert len(result) == 2
    left, right = sorted(result, key=lambda iv: iv.begin)
    assert left.begin == 1 and left.end == 3
    assert right.begin == 7 and right.end == 10


def test_remove_fully_overlapping_same_interval():
    a = Interval(1, 5)
    b = Interval(1, 5)
    result = a.remove(b)
    assert result == []


def test_get_overlap_full_containment():
    a = Interval(1, 10)
    b = Interval(3, 5)
    o = a._get_overlap(b)
    assert o == b


def test_get_overlap_b_begin_less_equal_a_begin():
    a = Interval(5, 15)
    b = Interval(3, 8)
    o = a._get_overlap(b)
    assert o.begin == a.begin  # 5
    assert o.end == b.end  # 8


def test_get_overlap_b_begin_greater_a_begin():
    a = Interval(3, 8)
    b = Interval(5, 15)
    o = a._get_overlap(b)
    assert o.begin == b.begin  # 5
    assert o.end == a.end  # 8


def test_combine_intervals_idempotent_if_same():
    a = Interval(1, 10)
    b = Interval(1, 10)
    c = combine_intervals(a, b)
    assert c == a


def test_interval_hashability():
    a = Interval(1, 5)
    b = Interval(3, 10)
    s = {a, b}
    assert len(s) == 2
    s.add(Interval(1, 5))
    assert len(s) == 2


def test_overlaps_reflexive_and_symmetric():
    a = Interval(1, 5)
    assert a.overlaps(a)
    b = Interval(4, 10)
    assert a.overlaps(b)
    assert b.overlaps(a)

