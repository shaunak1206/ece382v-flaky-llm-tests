from sys import maxsize

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_fields_and_unpacking():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    begin, end = iv
    assert (begin, end) == (2, 7)
    assert iv == Interval(2, 7)
    assert iv != Interval(2, 8)


def test_interval_len_is_width():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(5, 5)) == 0
    assert len(Interval(-4, 4)) == 8


def test_interval_ordering_uses_begin_only():
    a = Interval(1, 100)
    b = Interval(2, 3)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert not (b < a)
    assert Interval(1, 5) <= Interval(1, 9)
    assert Interval(1, 5) >= Interval(1, 9)


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert Interval(0, 10) in outer
    assert Interval(0, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_merges_to_bounding_interval():
    assert Interval(1, 3) + Interval(2, 8) == Interval(1, 8)
    assert Interval(5, 6) + Interval(0, 1) == Interval(0, 6)
    assert Interval(0, 10) + Interval(3, 4) == Interval(0, 10)


def test_interval_overlaps():
    assert Interval(1, 5).overlaps(Interval(3, 8))
    assert Interval(3, 8).overlaps(Interval(1, 5))
    assert Interval(1, 10).overlaps(Interval(4, 6))
    assert not Interval(1, 3).overlaps(Interval(5, 7))
    assert not Interval(1, 3).overlaps(Interval(3, 5))


def test_interval_touches():
    assert Interval(1, 3).touches(Interval(3, 5))
    assert Interval(3, 5).touches(Interval(1, 3))
    assert not Interval(1, 3).touches(Interval(4, 6))
    assert not Interval(1, 5).touches(Interval(3, 8))


def test_interval_remove_middle_splits_in_two():
    assert Interval(1, 10).remove(Interval(4, 6)) == [Interval(1, 4), Interval(6, 10)]


def test_interval_remove_partial_overlap_and_disjoint_and_covered():
    assert Interval(1, 5).remove(Interval(3, 8)) == [Interval(1, 3)]
    assert Interval(1, 3).remove(Interval(5, 9)) == [Interval(1, 3)]
    assert Interval(3, 5).remove(Interval(1, 10)) == []


def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)


def test_combine_intervals_disjoint_spans_gap():
    assert combine_intervals(Interval(1, 2), Interval(8, 9)) == Interval(1, 9)


def test_combine_intervals_nested_returns_outer():
    outer = Interval(0, 100)
    inner = Interval(20, 30)
    assert combine_intervals(outer, inner) == outer
    assert combine_intervals(inner, outer) == outer


def test_combine_intervals_is_commutative_and_idempotent():
    a = Interval(-5, 3)
    b = Interval(0, 12)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(-5, 12)
    assert combine_intervals(a, a) == a


def test_combine_intervals_floats_and_negatives():
    result = combine_intervals(Interval(-2.5, -1.0), Interval(-1.5, 4.25))
    assert result == Interval(-2.5, 4.25)
    assert isinstance(result, Interval)


def test_combine_intervals_matches_add_operator():
    a = Interval(2, 6)
    b = Interval(4, 11)
    assert combine_intervals(a, b) == a + b


def test_get_minimum_bounding_interval_overlapping():
    result = get_minimum_bounding_interval(Interval(1, 3), Interval(2, 5))
    assert result == Interval(1, 5)
    assert isinstance(result, Interval)


def test_get_minimum_bounding_interval_disjoint_ordered():
    assert get_minimum_bounding_interval(Interval(0, 1), Interval(9, 10)) == Interval(0, 10)


def test_get_minimum_bounding_interval_identical():
    assert get_minimum_bounding_interval(Interval(4, 8), Interval(4, 8)) == Interval(4, 8)


def test_combine_three_intervals_ordered_chain():
    result = combine_three_intervals(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert result == Interval(1, 6)
    assert isinstance(result, Interval)


def test_combine_three_intervals_overlapping():
    result = combine_three_intervals(Interval(0, 4), Interval(2, 6), Interval(5, 10))
    assert result == Interval(0, 10)
