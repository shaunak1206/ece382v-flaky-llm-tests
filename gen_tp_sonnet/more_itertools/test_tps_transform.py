from itertools import accumulate, count

import pytest

import more_itertools as mi


def test_substrings():
    assert [''.join(s) for s in mi.substrings("more")] == [
        "m", "o", "r", "e", "mo", "or", "re", "mor", "ore", "more",
    ]
    assert list(mi.substrings([0, 1, 2])) == [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]


def test_partitions():
    result = [[''.join(p) for p in part] for part in mi.partitions("abc")]
    assert result == [["abc"], ["a", "bc"], ["ab", "c"], ["a", "b", "c"]]


def test_distinct_permutations():
    assert sorted(mi.distinct_permutations([1, 0, 1])) == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert len(list(mi.distinct_permutations("aabb"))) == 6


def test_derangements_have_no_fixed_points():
    items = ["Alice", "Bob", "Carol", "Dave"]
    result = list(mi.derangements(items))
    assert len(result) == 9
    for d in result:
        assert all(a != b for a, b in zip(d, items))


def test_intersperse():
    assert list(mi.intersperse("!", [1, 2, 3, 4, 5])) == [1, "!", 2, "!", 3, "!", 4, "!", 5]
    assert list(mi.intersperse(None, [1, 2, 3, 4, 5], n=2)) == [1, 2, None, 3, 4, None, 5]


def test_unique_to_each():
    assert mi.unique_to_each({"A", "B"}, {"B", "C"}, {"B", "D"}) == [["A"], ["C"], ["D"]]


def test_interleave_stops_at_shortest():
    assert list(mi.interleave([1, 2, 3], [4, 5], [6, 7, 8])) == [1, 4, 6, 2, 5, 7]


def test_interleave_longest_skips_exhausted():
    assert list(mi.interleave_longest([1, 2, 3], [4, 5], [6, 7, 8])) == [1, 4, 6, 2, 5, 7, 3, 8]


def test_collapse():
    iterable = [(1, 2), ([3, 4], [[5], [6]])]
    assert list(mi.collapse(iterable)) == [1, 2, 3, 4, 5, 6]
    assert list(mi.collapse(["ab", ("cd", "ef"), ["gh", "ij"]], base_type=tuple)) == [
        "ab",
        ("cd", "ef"),
        "gh",
        "ij",
    ]
    nested = [("a", ["b"]), ("c", ["d"])]
    assert list(mi.collapse(nested)) == ["a", "b", "c", "d"]
    assert list(mi.collapse(nested, levels=1)) == ["a", ["b"], "c", ["d"]]


def test_always_iterable():
    assert list(mi.always_iterable((1, 2, 3))) == [1, 2, 3]
    assert list(mi.always_iterable(1)) == [1]
    assert list(mi.always_iterable(None)) == []
    assert list(mi.always_iterable("foo")) == ["foo"]
    assert list(mi.always_iterable({"a": 1}, base_type=dict)) == [{"a": 1}]


def test_adjacent():
    assert list(mi.adjacent(lambda x: x == 3, range(6))) == [
        (False, 0),
        (False, 1),
        (True, 2),
        (True, 3),
        (True, 4),
        (False, 5),
    ]
    assert list(mi.adjacent(lambda x: x == 3, range(6), distance=2)) == [
        (False, 0),
        (True, 1),
        (True, 2),
        (True, 3),
        (True, 4),
        (True, 5),
    ]


def test_mark_ends():
    assert list(mi.mark_ends("ABC")) == [
        (True, False, "A"),
        (False, False, "B"),
        (False, True, "C"),
    ]


def test_locate_and_rlocate():
    assert list(mi.locate([0, 1, 1, 0, 1, 0, 0])) == [1, 2, 4]
    assert list(mi.locate(["a", "b", "c", "b"], lambda x: x == "b")) == [1, 3]
    assert list(mi.rlocate([0, 1, 1, 0, 1, 0, 0])) == [4, 2, 1]
    assert list(mi.rlocate(iter("abcb"), lambda x: x == "b")) == [3, 1]


def test_longest_common_prefix():
    assert "".join(mi.longest_common_prefix(["abcd", "abc", "abf"])) == "ab"


def test_lstrip_rstrip_strip():
    iterable = (None, False, None, 1, 2, None, 3, False, None)
    pred = lambda x: x in {None, False, ""}  # noqa: E731
    assert list(mi.lstrip(iterable, pred)) == [1, 2, None, 3, False, None]
    assert list(mi.rstrip(iterable, pred)) == [None, False, None, 1, 2, None, 3]
    assert list(mi.strip(iterable, pred)) == [1, 2, None, 3]


def test_islice_extended():
    assert list(mi.islice_extended(iter("abcdefgh"), -4, -1)) == ["e", "f", "g"]
    assert list(mi.islice_extended(count(), 110, 99, -2)) == [110, 108, 106, 104, 102, 100]
    assert list(mi.islice_extended(map(str, count()))[10:20:2]) == ["10", "12", "14", "16", "18"]


def test_difference_inverts_accumulate():
    assert list(mi.difference(accumulate([0, 1, 2, 3, 4]))) == [0, 1, 2, 3, 4]
    assert list(mi.difference([1, 2, 6, 24, 120], lambda x, y: x // y)) == [1, 2, 3, 4, 5]


def test_seekable():
    it = mi.seekable(str(n) for n in count())
    assert (next(it), next(it), next(it)) == ("0", "1", "2")
    it.seek(0)
    assert (next(it), next(it), next(it)) == ("0", "1", "2")
    it2 = mi.seekable(str(n) for n in range(20))
    it2.seek(10)
    assert next(it2) == "10"
    it2.seek(20)
    assert list(it2) == []


def test_exactly_n():
    assert mi.exactly_n([True, True, False], 2) is True
    assert mi.exactly_n([True, True, False], 1) is False
    assert mi.exactly_n([0, 1, 2, 3, 4, 5], 3, lambda x: x < 3) is True


def test_circular_shifts():
    assert list(mi.circular_shifts(range(4))) == [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    assert list(mi.circular_shifts(range(4), 2)) == [(0, 1, 2, 3), (2, 3, 0, 1)]
    assert list(mi.circular_shifts(range(4), -1)) == [
        (0, 1, 2, 3),
        (3, 0, 1, 2),
        (2, 3, 0, 1),
        (1, 2, 3, 0),
    ]


def test_replace():
    pred = lambda x: x == 0  # noqa: E731
    assert list(mi.replace([1, 1, 0, 1, 1, 0, 1, 1], pred, (2, 3))) == [1, 1, 2, 3, 1, 1, 2, 3, 1, 1]
    assert list(mi.replace([1, 1, 0, 1, 1, 0, 1, 1, 0], pred, [None], count=2)) == [
        1, 1, None, 1, 1, None, 1, 1, 0,
    ]


def test_is_sorted():
    assert mi.is_sorted(["1", "2", "3", "4", "5"], key=int) is True
    assert mi.is_sorted([5, 4, 3, 1, 2], reverse=True) is False
    assert mi.is_sorted([1, 1, 2]) is True
    assert mi.is_sorted([1, 1, 2], strict=True) is False


def test_numeric_range():
    assert list(mi.numeric_range(3.5)) == [0.0, 1.0, 2.0, 3.0]
    r = mi.numeric_range(0, 10, 2)
    assert list(r) == [0, 2, 4, 6, 8]
    assert len(r) == 5
    assert r.index(4) == 2
    assert 6 in r
    assert 5 not in r


def test_iequals():
    assert mi.iequals("abc", ["a", "b", "c"], ("a", "b", "c"), iter("abc")) is True
    assert mi.iequals("abc", "acb") is False


def test_filter_except_and_map_except():
    iterable = ["1", "2", "three", "4", None]
    assert list(mi.filter_except(int, iterable, ValueError, TypeError)) == ["1", "2", "4"]
    assert list(mi.map_except(int, iterable, ValueError, TypeError)) == [1, 2, 4]
    with pytest.raises(TypeError):
        list(mi.map_except(int, ["1", None], ValueError))
