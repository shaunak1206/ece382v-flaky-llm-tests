from sys import maxsize

from intervaltree.interval import (
    Interval,
    combine_intervals,
    combine_three_intervals,
    get_minimum_bounding_interval,
)


def test_interval_fields():
    iv = Interval(2, 7)
    assert iv.begin == 2
    assert iv.end == 7
    assert tuple(iv) == (2, 7)


def test_interval_defaults():
    iv = Interval()
    assert iv.begin == 0
    assert iv.end == maxsize


def test_interval_len_is_width():
    assert len(Interval(3, 10)) == 7
    assert len(Interval(5, 5)) == 0


def test_interval_equality_and_hash():
    assert Interval(1, 4) == Interval(1, 4)
    assert Interval(1, 4) != Interval(1, 5)
    assert hash(Interval(1, 4)) == hash(Interval(1, 4))
    assert len({Interval(1, 4), Interval(1, 4)}) == 1


def test_interval_ordering_by_begin():
    a = Interval(1, 10)
    b = Interval(2, 3)
    assert a < b
    assert b > a
    assert a <= b
    assert b >= a
    assert a <= Interval(1, 2)
    assert a >= Interval(1, 2)


def test_interval_contains():
    outer = Interval(0, 10)
    assert Interval(2, 5) in outer
    assert outer in outer
    assert Interval(5, 11) not in outer
    assert Interval(-1, 5) not in outer


def test_interval_add_merges_overlapping():
    assert Interval(1, 5) + Interval(3, 8) == Interval(1, 8)
    assert Interval(3, 8) + Interval(1, 5) == Interval(1, 8)


def test_interval_add_disjoint_spans_gap():
    assert Interval(0, 2) + Interval(10, 12) == Interval(0, 12)


def test_interval_overlaps_and_touches():
    assert Interval(0, 5).overlaps(Interval(3, 8))
    assert not Interval(0, 5).overlaps(Interval(5, 8))
    assert not Interval(0, 5).overlaps(Interval(6, 8))
    assert Interval(0, 5).touches(Interval(5, 8))
    assert Interval(5, 8).touches(Interval(0, 5))
    assert not Interval(0, 5).touches(Interval(6, 8))


def test_interval_remove_middle_splits():
    result = Interval(0, 10).remove(Interval(3, 6))
    assert result == [Interval(0, 3), Interval(6, 10)]


def test_interval_remove_disjoint_returns_self():
    assert Interval(0, 5).remove(Interval(10, 12)) == [Interval(0, 5)]


def test_interval_remove_fully_covered_returns_empty():
    assert Interval(2, 4).remove(Interval(0, 10)) == []


def test_combine_intervals_overlapping():
    assert combine_intervals(Interval(1, 5), Interval(3, 9)) == Interval(1, 9)


def test_combine_intervals_is_symmetric():
    a = Interval(2, 6)
    b = Interval(4, 12)
    assert combine_intervals(a, b) == combine_intervals(b, a) == Interval(2, 12)


def test_combine_intervals_nested_and_disjoint():
    assert combine_intervals(Interval(0, 10), Interval(3, 4)) == Interval(0, 10)
    assert combine_intervals(Interval(0, 2), Interval(8, 9)) == Interval(0, 9)


def test_combine_intervals_identical():
    assert combine_intervals(Interval(3, 7), Interval(3, 7)) == Interval(3, 7)


def test_combine_intervals_returns_interval_type():
    assert isinstance(combine_intervals(Interval(1, 2), Interval(2, 3)), Interval)


def test_get_minimum_bounding_interval_ordered_inputs():
    result = get_minimum_bounding_interval(Interval(1, 4), Interval(6, 10))
    assert result == Interval(1, 10)
    assert isinstance(result, Interval)


def test_get_minimum_bounding_interval_same_interval():
    iv = Interval(3, 9)
    assert get_minimum_bounding_interval(iv, iv) == Interval(3, 9)


def test_get_minimum_bounding_interval_default_intervals():
    result = get_minimum_bounding_interval(Interval(), Interval())
    assert result == Interval(0, maxsize)


def test_combine_three_intervals_ordered_inputs():
    result = combine_three_intervals(Interval(0, 2), Interval(3, 5), Interval(6, 9))
    assert result == Interval(0, 9)
    assert isinstance(result, Interval)


def test_combine_three_intervals_identical():
    iv = Interval(4, 8)
    assert combine_three_intervals(iv, iv, iv) == Interval(4, 8)
