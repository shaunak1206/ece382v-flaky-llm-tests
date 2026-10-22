import more_itertools as mi


def test_split_at_basic():
    assert list(mi.split_at("abcdcba", lambda x: x == "b")) == [["a"], ["c", "d", "c"], ["a"]]
    assert list(mi.split_at(range(10), lambda n: n % 2 == 1)) == [[0], [2], [4], [6], [8], []]


def test_split_at_maxsplit():
    result = list(mi.split_at(range(10), lambda n: n % 2 == 1, maxsplit=2))
    assert result == [[0], [2], [4, 5, 6, 7, 8, 9]]


def test_split_before():
    assert list(mi.split_before("OneTwo", lambda s: s.isupper())) == [
        ["O", "n", "e"],
        ["T", "w", "o"],
    ]
    assert list(mi.split_before(range(10), lambda n: n % 3 == 0)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9],
    ]


def test_split_after():
    assert list(mi.split_after(range(10), lambda n: n % 3 == 0)) == [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert list(mi.split_after(range(10), lambda n: n % 3 == 0, maxsplit=2)) == [
        [0],
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
    ]


def test_split_when():
    result = list(mi.split_when([1, 2, 3, 3, 2, 5, 2, 4, 2], lambda x, y: x > y))
    assert result == [[1, 2, 3, 3], [2, 5], [2, 4], [2]]


def test_split_into():
    assert list(mi.split_into([1, 2, 3, 4, 5, 6], [1, 2, 3])) == [[1], [2, 3], [4, 5, 6]]
    assert list(mi.split_into([1, 2, 3, 4, 5, 6], [2, 3])) == [[1, 2], [3, 4, 5]]
    assert list(mi.split_into([1, 2, 3, 4], [1, 2, 3, 4])) == [[1], [2, 3], [4], []]


def test_padded_with_n():
    assert list(mi.padded([1, 2, 3], "?", 5)) == [1, 2, 3, "?", "?"]


def test_padded_next_multiple():
    assert list(mi.padded([1, 2, 3, 4], n=3, next_multiple=True)) == [1, 2, 3, 4, None, None]


def test_repeat_each():
    assert list(mi.repeat_each("ABC", 3)) == ["A", "A", "A", "B", "B", "B", "C", "C", "C"]


def test_repeat_last():
    from itertools import islice

    assert list(islice(mi.repeat_last(range(3)), 5)) == [0, 1, 2, 2, 2]
    assert list(islice(mi.repeat_last(range(0), 42), 5)) == [42, 42, 42, 42, 42]


def test_distribute_round_robin():
    children = mi.distribute(3, [1, 2, 3, 4, 5, 6, 7])
    assert [list(c) for c in children] == [[1, 4, 7], [2, 5], [3, 6]]


def test_divide_keeps_order():
    children = mi.divide(3, [1, 2, 3, 4, 5, 6, 7])
    assert [list(c) for c in children] == [[1, 2, 3], [4, 5], [6, 7]]


def test_unzip():
    letters, numbers = mi.unzip([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    assert list(letters) == ["a", "b", "c", "d"]
    assert list(numbers) == [1, 2, 3, 4]


def test_stagger():
    assert list(mi.stagger([0, 1, 2, 3])) == [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
    assert list(mi.stagger(range(8), offsets=(0, 2, 4))) == [
        (0, 2, 4),
        (1, 3, 5),
        (2, 4, 6),
        (3, 5, 7),
    ]


def test_zip_offset():
    assert list(mi.zip_offset("0123", "abcdef", offsets=(0, 1))) == [
        ("0", "b"),
        ("1", "c"),
        ("2", "d"),
        ("3", "e"),
    ]
    assert list(mi.zip_offset("0123", "abcdef", offsets=(0, 1), longest=True)) == [
        ("0", "b"),
        ("1", "c"),
        ("2", "d"),
        ("3", "e"),
        (None, "f"),
    ]


def test_sort_together():
    assert mi.sort_together([(4, 3, 2, 1), ("a", "b", "c", "d")]) == [
        (1, 2, 3, 4),
        ("d", "c", "b", "a"),
    ]
    iterables = [(3, 1, 2), (0, 1, 0), ("c", "b", "a")]
    assert mi.sort_together(iterables, key_list=(1, 2)) == [
        (2, 3, 1),
        (0, 0, 1),
        ("a", "c", "b"),
    ]


def test_windowed_basic():
    assert list(mi.windowed([1, 2, 3, 4, 5], 3)) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_windowed_fillvalue_and_step():
    assert list(mi.windowed([1, 2, 3], 4)) == [(1, 2, 3, None)]
    assert list(mi.windowed([1, 2, 3, 4, 5, 6], 3, fillvalue="!", step=2)) == [
        (1, 2, 3),
        (3, 4, 5),
        (5, 6, "!"),
    ]


def test_bucket():
    iterable = ["a1", "b1", "c1", "a2", "b2", "c2", "b3"]
    s = mi.bucket(iterable, key=lambda x: x[0])
    assert sorted(list(s)) == ["a", "b", "c"]
    assert list(s["b"]) == ["b1", "b2", "b3"]
    assert list(s["a"]) == ["a1", "a2"]


def test_groupby_transform():
    result = list(
        mi.groupby_transform(
            "aAAbBBcCC",
            lambda k: k.upper(),
            lambda v: v.lower(),
            lambda g: "".join(g),
        )
    )
    assert result == [("A", "aaa"), ("B", "bbb"), ("C", "ccc")]


def test_map_reduce():
    keyfunc = lambda x: x.upper()  # noqa: E731
    result = mi.map_reduce("abbccc", keyfunc)
    assert sorted(result.items()) == [("A", ["a"]), ("B", ["b", "b"]), ("C", ["c", "c", "c"])]
    counts = mi.map_reduce("abbccc", keyfunc, lambda x: 1, sum)
    assert dict(counts) == {"A": 1, "B": 2, "C": 3}


def test_run_length_encode_decode():
    encoded = list(mi.run_length.encode("abbcccdddd"))
    assert encoded == [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    assert "".join(mi.run_length.decode(encoded)) == "abbcccdddd"


def test_consecutive_groups():
    iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
    groups = [list(g) for g in mi.consecutive_groups(iterable)]
    assert groups == [[1], [10, 11, 12], [20], [30, 31, 32, 33], [40]]
