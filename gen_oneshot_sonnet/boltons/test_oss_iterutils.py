"""Tests for boltons.iterutils."""

import itertools
import re
import string

import pytest

from boltons.iterutils import (
    GUIDerator,
    PathAccessError,
    backoff,
    backoff_iter,
    bucketize,
    chunk_ranges,
    chunked,
    chunked_iter,
    first,
    flatten,
    flatten_iter,
    frange,
    get_path,
    guid_iter,
    is_collection,
    is_iterable,
    is_scalar,
    lstrip,
    one,
    pairwise,
    pairwise_iter,
    partition,
    redundant,
    remap,
    research,
    rstrip,
    same,
    seq_guid_iter,
    soft_sorted,
    split,
    split_iter,
    strip,
    unique,
    untyped_sorted,
    windowed,
    windowed_iter,
    xfrange,
)


def test_chunked_and_chunked_iter():
    assert chunked(range(10), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunked(range(10), 3, fill=None) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
    assert chunked(range(10), 3, count=2) == [[0, 1, 2], [3, 4, 5]]
    assert list(chunked_iter(range(6), 2)) == [[0, 1], [2, 3], [4, 5]]
    assert chunked((i for i in range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert chunked([], 3) == []


def test_chunked_strings_bytes_and_invalid_arguments():
    assert chunked("abcdefg", 3) == ["abc", "def", "g"]
    assert chunked(b"abcdef", 4) == [b"abcd", b"ef"]
    with pytest.raises(ValueError):
        chunked([1, 2, 3], 0)
    with pytest.raises(ValueError):
        chunked([1, 2, 3], -1)
    with pytest.raises(ValueError):
        chunked([1, 2, 3], 2, bogus=True)
    with pytest.raises(TypeError):
        chunked(5, 2)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(input_offset=10, input_size=10, chunk_size=5), [(10, 15), (15, 20)]),
        (dict(input_offset=10, input_size=10, chunk_size=5, overlap_size=1), [(10, 15), (14, 19), (18, 20)]),
        (dict(input_offset=10, input_size=10, chunk_size=5, overlap_size=2), [(10, 15), (13, 18), (16, 20)]),
        (dict(input_offset=4, input_size=15, chunk_size=5, align=False), [(4, 9), (9, 14), (14, 19)]),
        (dict(input_offset=4, input_size=15, chunk_size=5, align=True), [(4, 5), (5, 10), (10, 15), (15, 19)]),
        (
            dict(input_offset=2, input_size=15, chunk_size=5, overlap_size=1, align=False),
            [(2, 7), (6, 11), (10, 15), (14, 17)],
        ),
        (
            dict(input_offset=2, input_size=15, chunk_size=5, overlap_size=1, align=True),
            [(2, 5), (4, 9), (8, 13), (12, 17)],
        ),
        (
            dict(input_offset=3, input_size=15, chunk_size=5, overlap_size=1, align=True),
            [(3, 5), (4, 9), (8, 13), (12, 17), (16, 18)],
        ),
    ],
)
def test_chunk_ranges(kwargs, expected):
    assert list(chunk_ranges(**kwargs)) == expected


def test_windowed_and_pairwise():
    assert windowed(range(7), 3) == [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
    assert windowed(range(3), 5) == []
    assert windowed(range(4), 3, fill=None) == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, None),
        (3, None, None),
    ]
    assert list(windowed_iter("abc", 2)) == [("a", "b"), ("b", "c")]
    assert pairwise(range(5)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert pairwise([]) == []
    assert pairwise(range(3), end=None) == [(0, 1), (1, 2), (2, None)]
    assert list(pairwise_iter(range(3), end=None)) == [(0, 1), (1, 2), (2, None)]


def test_bucketize_variants_and_errors():
    assert bucketize(range(5)) == {False: [0], True: [1, 2, 3, 4]}
    assert bucketize(range(5), lambda x: x % 2 == 1) == {False: [0, 2, 4], True: [1, 3]}
    assert bucketize([1 + 1j, 2 + 2j, 1, 2], key="real") == {1.0: [(1 + 1j), 1], 2.0: [(2 + 2j), 2]}
    assert bucketize([1, 2, 365, 4, 98], key=[0, 1, 2, 0, 2]) == {0: [1, 4], 1: [2], 2: [365, 98]}
    assert bucketize([None, None, None, "hello"]) == {False: [None, None, None], True: ["hello"]}
    assert bucketize(range(10), lambda x: x % 3) == {0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}
    assert bucketize(range(5), value_transform=lambda x: x * x) == {False: [0], True: [1, 4, 9, 16]}
    assert bucketize(range(10), key=lambda x: x % 3, key_filter=lambda k: k % 3 != 1) == {
        0: [0, 3, 6, 9],
        2: [2, 5, 8],
    }
    with pytest.raises(TypeError):
        bucketize(5)
    with pytest.raises(ValueError):
        bucketize([1, 2], key=[0])
    with pytest.raises(TypeError):
        bucketize([1], key=5)


def test_partition():
    nonempty, empty = partition(["", "", "hi", "", "bye"])
    assert nonempty == ["hi", "bye"]
    assert empty == ["", "", ""]

    is_digit = lambda x: x in string.digits  # noqa: E731
    decimal_digits, hexletters = partition(string.hexdigits, is_digit)
    assert "".join(decimal_digits) == "0123456789"
    assert "".join(hexletters) == "abcdefABCDEF"

    assert partition(range(-1, 2), lambda i: i > 0, lambda i: i < 0) == ([1], [-1], [0])
    assert partition([1 + 2j, 3 + 0j], "imag") == ([1 + 2j], [3 + 0j])
    with pytest.raises(TypeError):
        partition(5)


def test_unique_and_redundant():
    assert "".join(unique("11010110001010010101010")) == "10"
    assert unique([1, 2, 3] * 10) == [1, 2, 3]
    assert unique(["hi", "hello", "ok", "bye", "yes"], key=len) == ["hi", "hello", "bye"]
    assert unique(["a", "b", "a"], key="no_such_attribute") == ["a", "b"]
    with pytest.raises(TypeError):
        unique(5)
    with pytest.raises(TypeError):
        unique([1], key=5)

    assert redundant([1, 2, 3, 4]) == []
    assert redundant([1, 2, 3, 2, 3, 3, 4]) == [2, 3]
    assert redundant([1, 2, 3, 2, 3, 3, 4], groups=True) == [[2, 2], [3, 3, 3]]
    assert redundant(["hi", "Hi", "HI", "hello"], key=str.lower) == ["Hi"]
    assert redundant(["hi", "Hi", "HI", "hello"], groups=True, key=str.lower) == [["hi", "Hi", "HI"]]


def test_first_and_one():
    assert first([0, False, None, [], (), 42]) == 42
    assert first([0, False, None, [], ()]) is None
    assert first([0, False, None, [], ()], default="ohai") == "ohai"
    assert first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0) == 4
    assert first(iter([])) is None
    match = first(re.match(regex, "abc") for regex in ["b.*", "a(.*)"])
    assert match.group(1) == "bc"

    assert one((True, False, False)) is True
    assert one((True, False, True)) is None
    assert one((0, 0, "a")) == "a"
    assert one((0, False, None)) is None
    assert one((True, True), default=False) is False
    assert one((10, 20, 30, 42), key=lambda i: i > 40) == 42


def test_predicates_flatten_and_same():
    assert is_iterable([])
    assert not is_iterable(object())
    assert is_scalar("hello") and is_scalar(object()) and not is_scalar(range(10))
    assert is_collection(range(10))
    assert not is_collection("hello")
    assert not is_collection(b"bytes")
    assert not is_collection(object())

    nested = [[1, 2], [[3], [4, 5]]]
    assert flatten(nested) == [1, 2, 3, 4, 5]
    assert list(flatten_iter(nested)) == [1, 2, 3, 4, 5]
    assert flatten(["ab", ["cd", ("e",)]]) == ["ab", "cd", "e"]

    assert same([]) and same([1]) and same(["a", "a", "a"])
    assert not same(range(20))
    assert same([[], []])
    assert not same([[], []], ref="test")
    assert same([2, 2], ref=2)


def test_split_and_strip_families():
    assert split(["hi", "hello", None, None, "sup", None, "soap", None]) == [["hi", "hello"], ["sup"], ["soap"]]
    assert list(split_iter(["hi", "hello", None, None, "sup", None], sep=[None])) == [
        ["hi", "hello"],
        [],
        ["sup"],
        [],
    ]
    falsy_sep = lambda x: not x  # noqa: E731
    assert list(split_iter(["hi", "hello", None, "", "sup", False], falsy_sep)) == [
        ["hi", "hello"],
        [],
        ["sup"],
        [],
    ]
    assert split([1, 0, 2, 0, 3], 0, maxsplit=1) == [[1], [2, 0, 3]]
    assert split([1, 2, 3], 2, maxsplit=0) == [[1, 2, 3]]
    with pytest.raises(TypeError):
        split(5)

    assert lstrip(["Foo", "Bar", "Bam"], "Foo") == ["Bar", "Bam"]
    assert rstrip(["Foo", "Bar", "Bam"], "Bam") == ["Foo", "Bar"]
    assert strip(["Fu", "Foo", "Bar", "Bam", "Fu"], "Fu") == ["Foo", "Bar", "Bam"]
    assert strip([None, 1, None]) == [1]
    assert rstrip([1, "x", 2, "x", "x"], "x") == [1, "x", 2]
    assert lstrip(["x", "x", 1, "x"], "x") == [1, "x"]


def test_frange_and_xfrange():
    assert frange(5) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frange(6, step=1.25) == [0.0, 1.25, 2.5, 3.75, 5.0]
    assert frange(100.5, 101.5, 0.25) == [100.5, 100.75, 101.0, 101.25]
    assert frange(5, 0) == []
    assert frange(5, 0, step=-1.25) == [5.0, 3.75, 2.5, 1.25]
    assert tuple(xfrange(1, 3, step=0.75)) == (1.0, 1.75, 2.5)
    with pytest.raises(ValueError):
        frange(5, step=0)


def test_backoff_sequences_and_validation():
    assert backoff(1, 10) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=5)) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=8)) == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    assert list(backoff_iter(0.25, 100.0, factor=10)) == [0.25, 2.5, 25.0, 100.0]
    assert list(itertools.islice(backoff_iter(1, 4, count="repeat"), 6)) == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]

    base = backoff(1, 10)
    jittered = backoff(1, 10, jitter=True)
    assert len(jittered) == len(base)
    assert all(0 <= j <= b for j, b in zip(jittered, base))

    with pytest.raises(ValueError):
        backoff(1, 10, count="repeat")
    with pytest.raises(ValueError):
        backoff(10, 1)
    with pytest.raises(ValueError):
        backoff(1, 10, factor=0.5)
    with pytest.raises(ValueError):
        backoff(-1, 10)
    with pytest.raises(ValueError):
        backoff(1, 0)


def test_remap_filter_transform_and_copy():
    reviews = {"Star Trek": {"TNG": 10, "DS9": 8.5, "ENT": None}, "Babylon 5": 6, "Dr. Who": None}
    assert remap(reviews, lambda p, k, v: v is not None) == {
        "Babylon 5": 6,
        "Star Trek": {"DS9": 8.5, "TNG": 10},
    }

    data = {"a": 1, "b": [1, 2, {"c": 3}], "t": (4, 5)}

    def double(path, key, value):
        if isinstance(value, int):
            return key, value * 2
        return key, value

    assert remap(data, visit=double) == {"a": 2, "b": [2, 4, {"c": 6}], "t": (8, 10)}

    upper = remap(
        {"a": {"b": 1}},
        visit=lambda p, k, v: (k.upper() if isinstance(k, str) else k, v),
    )
    assert upper == {"A": {"B": 1}}

    copied = remap(data)
    assert copied == data
    assert copied is not data
    assert copied["b"] is not data["b"]

    seen = []

    def spy(path, key, value):
        seen.append((path, key))
        return True

    remap({"a": {"b": 1}}, visit=spy)
    assert (("a",), "b") in seen
    assert ((), "a") in seen


def test_remap_self_reference_and_error_handling():
    looped = []
    looped.append(looped)
    result = remap(looped)
    assert result[0] is result

    with pytest.raises(TypeError):
        remap({}, visit="nope")
    with pytest.raises(TypeError):
        remap(5)
    with pytest.raises(TypeError):
        remap({}, bogus=1)

    def plus_one(path, key, value):
        return key, value + 1

    with pytest.raises(TypeError):
        remap({"a": "x"}, visit=plus_one)
    assert remap({"a": 1, "b": "x"}, visit=plus_one, reraise_visit=False) == {"a": 2, "b": "x"}


def test_research_and_get_path():
    root = {"a": {"b": 1, "c": (2, "d", 3)}, "e": None}
    res = research(root, query=lambda p, k, v: isinstance(v, int))
    assert sorted(res) == [(("a", "b"), 1), (("a", "c", 0), 2), (("a", "c", 2), 3)]

    nested = {"a": {"b": {"c": [[1], [2], [3]]}}}
    assert get_path(nested, ("a", "b", "c", 2, 0)) == 3
    assert get_path(nested, "a.b.c.2.0") == 3
    assert get_path(nested, ("a", "x"), default="dflt") == "dflt"
    with pytest.raises(PathAccessError):
        get_path(nested, ("a", "b", "c", 5))
    with pytest.raises(KeyError):
        get_path(nested, ("nope",))


def test_soft_sorted_and_untyped_sorted():
    assert soft_sorted(["two", "b", "one", "a"], first=["one", "two"]) == ["one", "two", "a", "b"]
    assert soft_sorted(range(7), first=[6, 15], last=[2, 4], reverse=True) == [6, 5, 3, 1, 0, 2, 4]
    assert "".join(soft_sorted(string.hexdigits, first="za1", last="b", key=str.lower)) == "aA1023456789cCdDeEfFbB"
    assert untyped_sorted(["abc", 2.0, 1, 2, "def"]) == [1, 2.0, 2, "abc", "def"]
    assert untyped_sorted([3, 1, 2], reverse=True) == [3, 2, 1]
    with pytest.raises(TypeError):
        untyped_sorted([1], key=5)


def test_guid_iterators():
    ids = [next(guid_iter) for _ in range(100)]
    assert len(set(ids)) == 100
    assert all(len(i) == 24 and int(i, 16) >= 0 for i in ids)

    assert len(next(GUIDerator(size=30))) == 30
    with pytest.raises(ValueError):
        GUIDerator(size=19)
    with pytest.raises(ValueError):
        GUIDerator(size=37)

    seq = [next(seq_guid_iter) for _ in range(50)]
    assert len(set(seq)) == 50
    assert seq == sorted(seq)
