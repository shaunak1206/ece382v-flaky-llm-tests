import pytest

from more_itertools import (
    bucket,
    partition,
    unique_everseen,
    unique_justseen,
    roundrobin,
    interleave_longest,
    nth,
    consume,
    first,
    last,
    one,
    only,
    always_iterable,
    split_at,
    split_before,
    collapse,
    run_length,
    map_reduce,
    distinct_permutations,
    powerset,
    nth_product,
)


def test_first_returns_first_item():
    assert first([3, 4, 5]) == 3


def test_first_empty_with_default():
    assert first([], "d") == "d"


def test_last_returns_last_item():
    assert last(range(5)) == 4


def test_last_empty_default():
    assert last([], default=None) is None


def test_one_single_item():
    assert one([42]) == 42


def test_one_too_many_raises_value_error():
    with pytest.raises(ValueError):
        one([1, 2])


def test_only_cases():
    assert only([]) is None
    assert only([], default="x") == "x"
    assert only([5]) == 5
    with pytest.raises(ValueError):
        only([1, 2])


def test_nth_basic_and_default():
    assert nth("abc", 1) == "b"
    assert nth("abc", 10) is None
    assert nth("abc", 10, default="z") == "z"


def test_consume_all():
    it = iter(range(5))
    consume(it)
    assert list(it) == []


def test_consume_n():
    it = iter(range(10))
    consume(it, 3)
    assert next(it) == 3


def test_partition_even_odd():
    evens_not, odds = partition(lambda x: x % 2, range(6))
    assert list(evens_not) == [0, 2, 4]
    assert list(odds) == [1, 3, 5]


def test_bucket_by_key():
    b = bucket([1, 2, 3, 11, 12, 21], key=lambda x: x // 10)
    assert list(b[0]) == [1, 2, 3]
    assert list(b[1]) == [11, 12]
    assert list(b[2]) == [21]


def test_bucket_missing_key_is_empty():
    b = bucket("aabbc", key=lambda c: c)
    assert list(b["z"]) == []
    assert "a" in b
    assert "z" not in b


def test_unique_everseen():
    assert list(unique_everseen("AAAABBBCCDAABBB")) == ["A", "B", "C", "D"]


def test_unique_justseen():
    assert list(unique_justseen("AAAABBBCCDAABBB")) == list("ABCDAB")


def test_roundrobin():
    assert list(roundrobin("ABC", "D", "EF")) == list("ADEBFC")


def test_interleave_longest():
    assert list(interleave_longest([1, 2, 3], [4, 5], [6])) == [1, 4, 6, 2, 5, 3]


def test_split_at_basic():
    assert list(split_at("abcdcba", lambda x: x == "b")) == [["a"], ["c", "d", "c"], ["a"]]


def test_split_at_maxsplit():
    assert list(split_at(range(10), lambda x: x % 3 == 0, maxsplit=1)) == [
        [],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    ]


def test_split_before():
    assert list(split_before("OneTwo", str.isupper)) == [["O", "n", "e"], ["T", "w", "o"]]


def test_collapse_nested_lists():
    assert list(collapse([1, [2, [3, [4]]], 5])) == [1, 2, 3, 4, 5]


def test_collapse_does_not_flatten_strings():
    assert list(collapse(["ab", ["cd"]])) == ["ab", "cd"]


def test_collapse_levels():
    assert list(collapse([[1], [[2]], [[[3]]]], levels=1)) == [1, [2], [[3]]]


def test_always_iterable_scalar_and_string():
    assert list(always_iterable(5)) == [5]
    assert list(always_iterable("abc")) == ["abc"]
    assert list(always_iterable(b"xy")) == [b"xy"]


def test_always_iterable_iterable_and_none():
    assert list(always_iterable([1, 2])) == [1, 2]
    assert list(always_iterable(None)) == []


def test_run_length_encode_decode():
    enc = list(run_length.encode("aaabccdddd"))
    assert enc == [("a", 3), ("b", 1), ("c", 2), ("d", 4)]
    assert "".join(run_length.decode(enc)) == "aaabccdddd"


def test_map_reduce_groups():
    result = map_reduce("abbccc", keyfunc=lambda c: c)
    assert dict(result) == {"a": ["a"], "b": ["b", "b"], "c": ["c", "c", "c"]}


def test_map_reduce_with_value_and_reduce():
    result = map_reduce(range(10), keyfunc=lambda x: x % 2, valuefunc=lambda x: x * 2, reducefunc=sum)
    assert dict(result) == {0: 40, 1: 50}


def test_distinct_permutations_with_repeats():
    assert sorted(distinct_permutations([1, 1, 2])) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]


def test_distinct_permutations_r():
    assert sorted(distinct_permutations("aab", 2)) == [("a", "a"), ("a", "b"), ("b", "a")]


def test_powerset():
    assert list(powerset([1, 2, 3])) == [
        (),
        (1,),
        (2,),
        (3,),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 2, 3),
    ]


def test_nth_product_matches_itertools_product():
    from itertools import product

    pools = ("ab", range(3), "xy")
    expected = list(product(*pools))
    for i in range(len(expected)):
        assert nth_product(i, *pools) == expected[i]


def test_nth_product_negative_and_out_of_range():
    assert nth_product(-1, range(2), range(2)) == (1, 1)
    with pytest.raises(IndexError):
        nth_product(4, range(2), range(2))
