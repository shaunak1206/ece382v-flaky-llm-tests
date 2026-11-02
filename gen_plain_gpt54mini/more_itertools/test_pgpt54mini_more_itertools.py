import math
import pickle
import random
import statistics
from collections import deque
from itertools import count, islice
from threading import Thread

import pytest

import more_itertools.more as mi
from more_itertools.recipes import _marker


def test_chunked_basic_and_strict():
    assert list(mi.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        list(mi.chunked([1, 2, 3], 2, strict=True))


def test_chunked_negative_n_raises():
    with pytest.raises(ValueError):
        list(mi.chunked([1, 2, 3], -1))


def test_first_and_last_and_nth_or_last():
    assert mi.first([9, 8, 7]) == 9
    assert mi.first([], default="x") == "x"
    assert mi.last([9, 8, 7]) == 7
    assert mi.last([], default="x") == "x"
    assert mi.nth_or_last([1, 2, 3, 4], 2) == 3
    assert mi.nth_or_last([1, 2], 5) == 2


def test_first_and_last_empty_raise_without_default():
    with pytest.raises(ValueError):
        mi.first([])
    with pytest.raises(ValueError):
        mi.last([])


def test_peekable_peek_prepend_and_getitem():
    p = mi.peekable(["a", "b", "c"])
    assert bool(p) is True
    assert p.peek() == "a"
    p.prepend("x", "y")
    assert next(p) == "x"
    assert p[0] == "y"
    assert p[1] == "a"
    assert list(p) == ["y", "a", "b", "c"]
    assert bool(p) is False


def test_peekable_negative_index_caches_remaining():
    p = mi.peekable([1, 2, 3])
    assert p[-1] == 3
    assert list(p) == [1, 2, 3]


def test_consumer_decorator_advances_generator():
    @mi.consumer
    def gen():
        received = yield
        yield received

    g = gen()
    assert next(g) is None
    assert g.send("hello") == "hello"


def test_ilen_consumes_iterable():
    it = iter([1, 2, 3, 4])
    assert mi.ilen(it) == 4
    assert list(it) == []


def test_iterate_and_takewhile_inclusive():
    assert list(islice(mi.iterate(lambda x: x * 2, 1), 5)) == [1, 2, 4, 8, 16]
    assert list(mi.takewhile_inclusive(lambda x: x < 5, [1, 4, 6, 4])) == [1, 4, 6]


def test_with_iter_closes_file_like_object():
    class CM:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return iter([1, 2, 3])

        def __exit__(self, exc_type, exc, tb):
            self.closed = True

    cm = CM()
    assert list(mi.with_iter(cm)) == [1, 2, 3]
    assert cm.closed is True


def test_sized_iterator_len_and_iteration():
    it = mi.sized_iterator(map(str, range(3)), 3)
    assert len(it) == 3
    assert list(it) == ["0", "1", "2"]


def test_one_and_only():
    assert mi.one([42]) == 42
    assert mi.only([42]) == 42
    assert mi.only([], default="missing") == "missing"


def test_one_and_only_too_many_raise():
    with pytest.raises(ValueError):
        mi.one([1, 2])
    with pytest.raises(TypeError):
        mi.one([1, 2], too_long=TypeError)
    with pytest.raises(ValueError):
        mi.only([1, 2])


def test_strictly_n_exact_short_and_long():
    assert list(mi.strictly_n([1, 2, 3], 3)) == [1, 2, 3]
    with pytest.raises(ValueError):
        list(mi.strictly_n([1, 2], 3))
    with pytest.raises(ValueError):
        list(mi.strictly_n([1, 2, 3, 4], 3))


def test_distinct_permutations_basic_and_r():
    assert sorted(mi.distinct_permutations([1, 0, 1])) == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert sorted(mi.distinct_permutations([1, 0, 1], r=2)) == [(0, 1), (1, 0), (1, 1)]


def test_derangements_basic_and_r():
    assert sorted(mi.derangements([0, 1, 2])) == [(1, 2, 0), (2, 0, 1)]
    assert sorted(mi.derangements([0, 1, 2], 2)) == [(1, 0), (2, 0)]


def test_intersperse_and_windowed():
    assert list(mi.intersperse("!", [1, 2, 3])) == [1, "!", 2, "!", 3]
    assert list(mi.intersperse(None, [1, 2, 3, 4], n=2)) == [1, 2, None, 3, 4]
    assert list(mi.windowed([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]


def test_windowed_input_shorter_than_n():
    assert list(mi.windowed([1, 2], 3, fillvalue="x")) == [(1, 2, "x")]


def test_windowed_invalid_params():
    with pytest.raises(ValueError):
        list(mi.windowed([1, 2, 3], 0))
    with pytest.raises(ValueError):
        list(mi.windowed([1, 2, 3], 2, step=0))


def test_unique_to_each_and_always_iterable():
    assert mi.unique_to_each({"A", "B"}, {"B", "C"}, {"B", "D"}) == [["A"], ["C"], ["D"]]
    assert list(mi.always_iterable(None)) == []
    assert list(mi.always_iterable("foo")) == ["foo"]
    assert list(mi.always_iterable({"a": 1}, base_type=dict)) == [{"a": 1}]


def test_always_iterable_none_base_type_none():
    assert list(mi.always_iterable("foo", base_type=None)) == ["f", "o", "o"]


def test_substrings_and_indexes():
    assert list(mi.substrings("abc")) == ["a", "b", "c", "ab", "bc", "abc"]
    assert list(mi.substrings_indexes("abc")) == [("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("ab", 0, 2), ("bc", 1, 3), ("abc", 0, 3)]
    assert list(mi.substrings_indexes("abc", reverse=True))[0] == ("abc", 0, 3)


def test_bucket_and_map_reduce():
    b = mi.bucket(["a1", "b1", "a2", "b2"], key=lambda x: x[0])
    assert sorted(list(b)) == ["a", "b"]
    assert list(b["a"]) == ["a1", "a2"]
    assert mi.map_reduce("abbccc", keyfunc=str.upper) == {"A": ["a"], "B": ["b", "b"], "C": ["c", "c", "c"]}


def test_spy_and_interleave_variants():
    head, it = mi.spy("abcdef", 2)
    assert head == ["a", "b"]
    assert list(it) == list("abcdef")
    assert list(mi.interleave([1, 2, 3], ["a", "b"], [True, False, None])) == [1, "a", True, 2, "b", False]
    assert list(mi.interleave_longest([1, 2, 3], ["a", "b"], [True, False, None])) == [1, "a", True, 2, "b", False, 3, None]


def test_interleave_evenly_basic_and_lengths_error():
    assert list(mi.interleave_evenly([[1, 2, 3], ["a", "b"]])) == [1, 2, "a", 3, "b"]
    with pytest.raises(ValueError):
        list(mi.interleave_evenly([[1, 2], [3]], lengths=[2]))


def test_interleave_randomly_deterministic_with_seed(monkeypatch):
    seq = [0, 0, 0, 0]

    def fake_randrange(n):
        return 0

    monkeypatch.setattr(mi, "randrange", fake_randrange)
    assert list(mi.interleave_randomly([1, 2], ["a", "b"])) == [1, 2, "a", "b"]


def test_collapse_basic_and_levels():
    assert list(mi.collapse([(1, 2), ([3, 4], [[5], [6]])])) == [1, 2, 3, 4, 5, 6]
    assert list(mi.collapse([("a", ["b"]), ("c", ["d"])], levels=1)) == ["a", ["b"], "c", ["d"]]


def test_side_effect_before_after_and_chunk_size():
    seen = []
    side = []

    def func(x):
        seen.append(x)

    def before():
        side.append("before")

    def after():
        side.append("after")

    assert list(mi.side_effect(func, [1, 2, 3], before=before, after=after)) == [1, 2, 3]
    assert seen == [1, 2, 3]
    assert side == ["before", "after"]

    chunks = []
    assert list(mi.side_effect(lambda c: chunks.append(tuple(c)), [1, 2, 3, 4], chunk_size=2)) == [1, 2, 3, 4]
    assert chunks == [(1, 2), (3, 4)]


def test_sliced_basic_and_strict():
    assert list(mi.sliced((1, 2, 3, 4, 5), 2)) == [(1, 2), (3, 4), (5,)]
    with pytest.raises(ValueError):
        list(mi.sliced((1, 2, 3), -1))
    with pytest.raises(ValueError):
        list(mi.sliced((1, 2, 3, 4, 5), 2, strict=True))


def test_split_at_before_after_when():
    assert list(mi.split_at("abcdcba", lambda x: x == "b")) == [["a"], ["c", "d", "c"], ["a"]]
    assert list(mi.split_before("OneTwo", lambda s: s.isupper())) == [["O", "n", "e"], ["T", "w", "o"]]
    assert list(mi.split_after("one1two2", lambda s: s.isdigit())) == [["o", "n", "e", "1"], ["t", "w", "o", "2"]]
    assert list(mi.split_when([1, 2, 3, 2, 5], lambda x, y: x > y)) == [[1, 2, 3], [2, 5]]


def test_split_into_and_padded():
    assert list(mi.split_into([1, 2, 3, 4, 5], [2, 2, None])) == [[1, 2], [3, 4], [5]]
    assert list(islice(mi.padded([1, 2, 3], "x"), 5)) == [1, 2, 3, "x", "x"]
    assert list(mi.repeat_each("ab", 3)) == ["a", "a", "a", "b", "b", "b"]
    assert list(islice(mi.repeat_last([1, 2]), 4)) == [1, 2, 2, 2]


def test_distribute_and_divide():
    groups = mi.distribute(3, [1, 2, 3, 4, 5, 6, 7])
    assert [list(g) for g in groups] == [[1, 4, 7], [2, 5], [3, 6]]
    assert [list(g) for g in mi.divide(3, [1, 2, 3, 4, 5, 6, 7])] == [[1, 2, 3], [4, 5], [6, 7]]


def test_stagger_and_zip_offset():
    assert list(mi.stagger([0, 1, 2, 3])) == [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
    assert list(mi.zip_offset("0123", "abcdef", offsets=(0, 1))) == [("0", "b"), ("1", "c"), ("2", "d"), ("3", "e")]
    with pytest.raises(ValueError):
        list(mi.zip_offset("a", "b", offsets=(0,)))


def test_sort_together_and_unzip():
    assert mi.sort_together([(4, 3, 2, 1), ("a", "b", "c", "d")]) == [(1, 2, 3, 4), ("d", "c", "b", "a")]
    letters, numbers = mi.unzip([("a", 1), ("b", 2)])
    assert list(letters) == ["a", "b"]
    assert list(numbers) == [1, 2]


def test_always_reversible_and_consecutive_groups():
    assert list(mi.always_reversible([1, 2, 3])) == [3, 2, 1]
    assert [list(g) for g in mi.consecutive_groups([1, 2, 11, 12, 21, 22])] == [[1, 2], [11, 12], [21, 22]]


def test_difference_and_groupby_transform():
    from itertools import accumulate

    assert list(mi.difference(accumulate([0, 1, 2, 3, 4]))) == [0, 1, 2, 3, 4]
    out = list(mi.groupby_transform("aAAbBB", keyfunc=str.upper, valuefunc=str.lower, reducefunc="".join))
    assert out == [("A", "aaa"), ("B", "bbb")]


def test_numeric_range_basics_and_indexing():
    r = mi.numeric_range(0, 5)
    assert list(r) == [0, 1, 2, 3, 4]
    assert len(r) == 5
    assert r[2] == 2
    assert r[-1] == 4
    assert r[1:4] == mi.numeric_range(1, 4)
    assert 3 in r
    assert r.index(3) == 3
    assert r.count(3) == 1


def test_numeric_range_zero_step_and_pickle_roundtrip():
    with pytest.raises(ValueError):
        mi.numeric_range(1, 5, 0)
    r = mi.numeric_range(1, 6, 2)
    assert pickle.loads(pickle.dumps(r)) == r


def test_count_cycle_and_mark_ends():
    assert list(mi.count_cycle("AB", 3)) == [(0, "A"), (0, "B"), (1, "A"), (1, "B"), (2, "A"), (2, "B")]
    assert list(mi.mark_ends("ABC")) == [(True, False, "A"), (False, False, "B"), (False, True, "C")]


def test_locate_and_rlocate():
    assert list(mi.locate([0, 1, 1, 0, 1, 0])) == [1, 2, 4]
    assert list(mi.rlocate([0, 1, 1, 0, 1, 0])) == [4, 2, 1]
    assert list(mi.locate("abcabc", lambda x: x == "b")) == [1, 4]
    assert list(mi.rlocate("abcabc", lambda x: x == "b")) == [4, 1]


def test_longest_common_prefix_lstrip_rstrip_strip():
    assert "".join(mi.longest_common_prefix(["abcd", "abc", "abf"])) == "ab"
    pred = lambda x: x in {None, False, ""}
    seq = (None, False, None, 1, 2, None, 3, False, None)
    assert list(mi.lstrip(seq, pred)) == [1, 2, None, 3, False, None]
    assert list(mi.rstrip(seq, pred)) == [None, False, None, 1, 2, None, 3]
    assert list(mi.strip(seq, pred)) == [1, 2, None, 3]


def test_islice_extended_negative_slices():
    assert list(mi.islice_extended("abcdefgh", -4, -1)) == ["e", "f", "g"]
    assert list(mi.islice_extended(count(), 110, 99, -2)) == [110, 108, 106, 104, 102, 100]


def test_seekable_peek_seek_relative_and_elements():
    it = mi.seekable((str(n) for n in range(10)))
    assert next(it) == "0"
    assert next(it) == "1"
    assert it.peek() == "2"
    assert it.elements()[:2] == ["0", "1"]
    it.seek(0)
    assert next(it) == "0"
    it.relative_seek(2)
    assert next(it) == "3"


def test_run_length_encode_decode():
    encoded = list(mi.run_length.encode("abbcccdddd"))
    assert encoded == [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    assert list(mi.run_length.decode(encoded)) == list("abbcccdddd")


def test_exactly_n_and_all_unique():
    assert mi.exactly_n([True, True, False], 2) is True
    assert mi.exactly_n([True, True, False], 1) is False
    assert mi.all_unique("ABCB") is False
    assert mi.all_unique("ABCb", str.lower) is False


def test_circular_shifts_and_gray_product():
    assert list(mi.circular_shifts(range(4))) == [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    assert list(mi.gray_product("AB", "CD")) == [("A", "C"), ("B", "C"), ("B", "D"), ("A", "D")]


def test_circular_shifts_invalid_steps():
    with pytest.raises(ValueError):
        list(mi.circular_shifts([1, 2, 3], 0))


def test_partial_product_and_outer_product():
    assert list(mi.partial_product("AB", "C", "DEF")) == [("A", "C", "D"), ("B", "C", "D"), ("B", "C", "E"), ("B", "C", "F")]
    assert list(mi.outer_product(lambda x, y: x + y, [1, 2], [10, 20])) == [(11, 21), (12, 22)]


def test_replace_and_filter_map_map_if():
    assert list(mi.replace([1, 0, 1, 0], lambda x: x == 0, [2, 3])) == [1, 2, 3, 1, 2, 3]
    assert list(mi.filter_map(lambda s: int(s) if s.isdigit() else None, ["1", "a", "2"])) == [1, 2]
    assert list(mi.map_if([0, 1, 2], lambda x: x > 0, lambda x: x * 10)) == [0, 10, 20]


def test_filter_except_and_map_except():
    assert list(mi.filter_except(int, ["1", "x", "2"], ValueError)) == ["1", "2"]
    assert list(mi.map_except(int, ["1", "x", "2"], ValueError)) == [1, 2]


def test_partitions_and_set_partitions():
    parts = list(mi.partitions("abc"))
    assert ["".join(p) for p in parts[0]] == ["abc"]
    assert ["".join(p) for p in parts[-1]] == ["a", "b", "c"]
    assert list(mi.set_partitions("abc", 2)) == [[["a"], ["b", "c"]], [["b"], ["a", "c"]], [["c"], ["a", "b"]]]


def test_time_limited_and_limit_zero():
    def gen():
        yield 1
        yield 2

    t = mi.time_limited(1, gen())
    assert list(t) == [1, 2]
    assert t.timed_out is False
    zero = mi.time_limited(0, [1, 2, 3])
    assert list(zero) == []
    assert zero.timed_out is True


def test_time_limited_negative_raises():
    with pytest.raises(ValueError):
        mi.time_limited(-1, [1])


def test_is_sorted_and_iequals():
    assert mi.is_sorted([1, 2, 3]) is True
    assert mi.is_sorted([1, 2, 2], strict=True) is False
    assert mi.iequals("abc", ["a", "b", "c"], iter("abc")) is True
    assert mi.iequals("abc", "acb") is False


def test_constrained_batches_and_errors():
    data = [b"12345", b"123", b"12345678", b"1", b"1", b"12", b"1"]
    assert list(mi.constrained_batches(data, 10)) == [(b"12345", b"123"), (b"12345678", b"1", b"1"), (b"12", b"1")]
    with pytest.raises(ValueError):
        list(mi.constrained_batches(data, 0))
    with pytest.raises(ValueError):
        list(mi.constrained_batches([b"123456"], 3, strict=True))


def test_join_mappings_and_value_chain():
    assert mi.join_mappings(score={"a": 1}, time={"a": 2}) == {"a": {"score": 1, "time": 2}}
    assert list(mi.value_chain(1, [2, 3], "x", b"y")) == [1, 2, 3, "x", b"y"]


def test_nth_product_permutation_combination_indexes():
    assert mi.nth_product(3, range(2), range(2)) == (1, 1)
    assert mi.nth_permutation("abcd", 2, 3) == ("b", "d")
    assert mi.nth_combination("abcd", 2, 3) == ("b", "d")
    assert mi.product_index([1, 1], range(2), range(2)) == 3
    assert mi.permutation_index([1, 3, 2], range(5)) == 19
    assert mi.combination_index("adf", "abcdefg") == 10
    assert mi.combination_with_replacement_index("adf", "abcdefg") == 20


def test_nth_combination_with_replacement_and_errors():
    assert mi.nth_combination_with_replacement(range(5), 3, 5) == (0, 1, 1)
    with pytest.raises(IndexError):
        mi.nth_product(100, range(2), range(2))
    with pytest.raises(IndexError):
        mi.nth_permutation("ab", 2, 10)
    with pytest.raises(IndexError):
        mi.nth_combination_with_replacement("ab", 2, 10)


def test_subfactorial_and_random_ordered_range(monkeypatch):
    assert mi.subfactorial(0) == 1
    assert mi.subfactorial(5) == 44
    monkeypatch.setattr(mi, "_random_ordered_indices", lambda n: range(n))
    assert list(mi.random_ordered_range(5)) == [0, 1, 2, 3, 4]


def test_unique_in_window_and_duplicates():
    assert list(mi.unique_in_window([0, 1, 0, 2, 3, 0], 3)) == [0, 1, 2, 3, 0]
    assert list(mi.duplicates_everseen("mississippi")) == ["s", "i", "s", "s", "i", "p", "i"]
    assert list(mi.duplicates_justseen("mississippi")) == ["s", "s", "p"]


def test_classify_unique_and_minmax():
    assert list(mi.classify_unique("otto")) == [("o", True, True), ("t", True, True), ("t", False, False), ("o", True, False)]
    assert mi.minmax([3, 1, 5]) == (1, 5)
    assert mi.minmax(4, 2, 6) == (2, 6)
    assert mi.minmax([], default=(0, 0)) == (0, 0)


def test_minmax_empty_raises_without_default():
    with pytest.raises(ValueError):
        mi.minmax([])


def test_all_unique_hashable_and_unhashable_mix():
    assert mi.all_unique([1, (1, 2), [3]], key=lambda x: repr(x)) is True


def test_countable_tracks_items_seen():
    c = mi.countable(iter(range(4)))
    assert c.items_seen == 0
    assert next(c) == 0
    assert next(c) == 1
    assert list(c) == [2, 3]
    assert c.items_seen == 4


def test_serialize_wraps_iterator_and_methods():
    s = mi.serialize(iter([1, 2, 3]))
    assert next(s) == 1
    assert list(s) == [2, 3]


def test_synchronized_wrapper_serializes_iterator():
    @mi.synchronized
    def gen():
        yield from [1, 2, 3]

    assert list(gen()) == [1, 2, 3]


def test_concurrent_tee_basic():
    a, b = mi.concurrent_tee([1, 2, 3], 2)
    assert next(a) == 1
    assert next(b) == 1
    assert list(a) == [2, 3]
    assert list(b) == [2, 3]


def test_concurrent_tee_invalid_n():
    with pytest.raises(ValueError):
        mi.concurrent_tee([1, 2, 3], -1)


def test_abortthread_is_baseexception():
    assert issubclass(mi.AbortThread, BaseException)


def test_run_length_encode_empty():
    assert list(mi.run_length.encode([])) == []


def test_islice_extended_getitem_requires_slice():
    it = mi.islice_extended(range(10))
    with pytest.raises(TypeError):
        _ = it[1]
