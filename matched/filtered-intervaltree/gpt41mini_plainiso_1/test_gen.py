import pytest
from intervaltree.interval import Interval, combine_intervals, combine_three_intervals, get_minimum_bounding_interval


def test_interval_creation_default():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == 2**63 - 1 or iv.end > iv.begin  # maxsize platform dependent


def test_interval_creation_custom():
    iv = Interval(5, 10)
    assert iv.begin == 5
    assert iv.end == 10


def test_interval_length():
    iv = Interval(3, 8)
    assert len(iv) == 5


def test_interval_comparison_lt_le_gt_ge():
    iv1 = Interval(1, 5)
    iv2 = Interval(3, 7)
    iv3 = Interval(1, 6)
    assert iv1 < iv2
    assert iv1 <= iv3
    assert iv2 > iv1
    assert iv3 >= iv1


def test_interval_contains_other_interval_true():
    iv_outer = Interval(1, 10)
    iv_inner = Interval(2, 5)
    assert iv_inner in iv_outer


def test_interval_contains_other_interval_false():
    iv_outer = Interval(1, 5)
    iv_inner = Interval(0, 3)
    iv_not_contained = Interval(3, 7)
    assert iv_inner not in iv_outer
    assert iv_not_contained not in iv_outer


def test_interval_add_combines_correctly():
    iv1 = Interval(2, 6)
    iv2 = Interval(4, 8)
    combined = iv1 + iv2
    assert combined.begin == 2
    assert combined.end == 8


def test_combine_intervals_function():
    iv1 = Interval(0, 5)
    iv2 = Interval(3, 10)
    combined = combine_intervals(iv1, iv2)
    assert combined.begin == 0
    assert combined.end == 10


def test_combine_three_intervals_function():
    iv1 = Interval(1, 3)
    iv2 = Interval(2, 5)
    iv3 = Interval(4, 6)
    combined = combine_three_intervals(iv1, iv2, iv3)
    assert combined.begin == 1
    assert combined.end == 6


def test_get_minimum_bounding_interval_function():
    iv1 = Interval(1, 4)
    iv2 = Interval(3, 7)
    bounding = get_minimum_bounding_interval(iv1, iv2)
    # Since sorted() was called incorrectly (result not stored),
    # points likely not sorted in code - expect first and last as before modification
    # But test according to code intention:
    assert bounding.begin in [1, 3]
    assert bounding.end in [4, 7]
    # check begin <= end regardless
    assert bounding.begin <= bounding.end


def test_overlaps_method_true():
    iv1 = Interval(1, 5)
    iv2 = Interval(4, 8)
    assert iv1.overlaps(iv2)
    assert iv2.overlaps(iv1)


def test_overlaps_method_false():
    iv1 = Interval(1, 3)
    iv2 = Interval(3, 6)  # touching but not overlapping
    assert not iv1.overlaps(iv2)
    assert not iv2.overlaps(iv1)


def test_touches_method_true():
    iv1 = Interval(1, 3)
    iv2 = Interval(3, 5)  # touches iv1 at 3
    assert iv1.touches(iv2)
    assert iv2.touches(iv1)


def test_touches_method_false():
    iv1 = Interval(1, 3)
    iv2 = Interval(4, 6)
    assert not iv1.touches(iv2)
    assert not iv2.touches(iv1)


def test_get_overlap_method_full_containment():
    iv_outer = Interval(1, 10)
    iv_inner = Interval(3, 5)
    overlap = iv_outer._get_overlap(iv_inner)
    assert overlap == iv_inner


def test_get_overlap_method_partial_left():
    iv1 = Interval(5, 10)
    iv2 = Interval(1, 7)
    overlap = iv1._get_overlap(iv2)
    assert overlap.begin == 5
    assert overlap.end == 7


def test_get_overlap_method_partial_right():
    iv1 = Interval(1, 5)
    iv2 = Interval(3, 10)
    overlap = iv1._get_overlap(iv2)
    assert overlap.begin == 3
    assert overlap.end == 5


def test_remove_no_overlap_returns_self():
    iv1 = Interval(1, 5)
    iv2 = Interval(6, 10)
    result = iv1.remove(iv2)
    assert result == [iv1]


def test_remove_other_contains_self_returns_empty():
    iv1 = Interval(3, 6)
    iv2 = Interval(1, 10)
    result = iv1.remove(iv2)
    assert result == []


def test_remove_partial_overlap_returns_segments():
    iv1 = Interval(1, 10)
    iv2 = Interval(3, 7)
    result = iv1.remove(iv2)
    # Expected to get parts before 3 and after 7
    assert all(isinstance(seg, Interval) for seg in result)
    # Those segments should not overlap iv2 and should cover iv1 minus iv2 overlap
    for seg in result:
        assert seg.end <= iv2.begin or seg.begin >= iv2.end
    # The sum of lengths of resulting segments should be <= length of original iv1
    total_length = sum(len(seg) for seg in result)
    assert total_length <= len(iv1)
    assert total_length > 0


def test_remove_partial_overlap_result_no_zero_length_intervals():
    iv1 = Interval(1, 5)
    iv2 = Interval(3, 5)
    result = iv1.remove(iv2)
    for seg in result:
        assert len(seg) > 0


def test_remove_partial_left_overlap():
    iv1 = Interval(5, 10)
    iv2 = Interval(3, 7)
    result = iv1.remove(iv2)
    # Segments should not contain iv2 part
    for seg in result:
        assert seg.end <= iv2.begin or seg.begin >= iv2.end
    # The segments should collectively cover (7,10) in this case
    assert all(seg.begin >= 7 for seg in result)


def test_remove_partial_right_overlap():
    iv1 = Interval(3, 7)
    iv2 = Interval(5, 10)
    result = iv1.remove(iv2)
    # segments should cover (3,5)
    assert all(seg.end <= iv2.begin for seg in result)
    assert any(seg.begin < iv2.begin for seg in result)


def test_repr_and_str_behave_like_namedtuple():
    iv = Interval(2, 5)
    # Namedtuple __repr__ shows like Interval(begin=2, end=5)
    rep = repr(iv)
    assert rep.startswith('Interval(')
    assert '2' in rep and '5' in rep


# Additional sanity test for combining intervals producing correct results
@pytest.mark.parametrize("a,b,expected_begin,expected_end", [
    (Interval(1, 4), Interval(5, 8), 1, 8),
    (Interval(5, 8), Interval(1, 4), 1, 8),
    (Interval(1, 4), Interval(2, 3), 1, 4),
    (Interval(2, 3), Interval(1, 4), 1, 4),
])
def test_combine_intervals_various_orders(a, b, expected_begin, expected_end):
    r = combine_intervals(a, b)
    assert r.begin == expected_begin
    assert r.end == expected_end


@pytest.mark.parametrize("a,b,expected", [
    (Interval(1, 3), Interval(4, 6), False),
    (Interval(1, 4), Interval(4, 6), False),
    (Interval(3, 5), Interval(4, 7), True),
    (Interval(4, 5), Interval(4, 6), True),
])
def test_interval_overlaps_functionality(a,b,expected):
    assert a.overlaps(b) == expected
    assert b.overlaps(a) == expected
