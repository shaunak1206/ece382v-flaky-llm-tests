import math
import pickle
from operator import add, mul

import pytest

from toolz.itertoolz import (
    accumulate,
    remove,
    groupby,
    merge_sorted,
    interleave,
    unique,
    isiterable,
    isdistinct,
    take,
    drop,
    take_nth,
    first,
    second,
    nth,
    last,
    get,
    concat,
    concatv,
    mapcat,
    cons,
    interpose,
    frequencies,
    reduceby,
    iterate,
    sliding_window,
    partition,
    partition_all,
    count,
    pluck,
    join,
    tail,
    diff,
    topk,
    peek,
    peekn,
    random_sample,
    rest,
    getter,
    no_pad,
)
from toolz.functoolz import (
    identity,
    apply,
    thread_first,
    thread_last,
    compose,
    compose_left,
    pipe,
    complement,
    juxt,
    do,
    flip,
    excepts,
    curry,
    memoize,
    is_arity,
    num_required_args,
    has_varargs,
    has_keywords,
    is_valid_args,
    is_partial_args,
)
from toolz.dicttoolz import (
    merge,
    merge_with,
    valmap,
    keymap,
    itemmap,
    valfilter,
    keyfilter,
    itemfilter,
    assoc,
    dissoc,
    assoc_in,
    update_in,
    get_in,
)
from toolz.recipes import countby, partitionby
from toolz.sandbox.core import EqualityHashKey, unzip
from toolz.sandbox.parallel import fold


def test_remove_filters_items():
    assert list(remove(lambda x: x % 2 == 0, [1, 2, 3, 4])) == [1, 3]


def test_accumulate_without_initial():
    assert list(accumulate(add, [1, 2, 3, 4])) == [1, 3, 6, 10]


def test_accumulate_with_initial_and_empty():
    assert list(accumulate(add, [], 1)) == [1]


def test_groupby_with_callable_key():
    out = groupby(len, ["a", "bb", "c", "dd"])
    assert out == {1: ["a", "c"], 2: ["bb", "dd"]}


def test_groupby_with_string_key():
    seq = [{"gender": "F", "name": "Alice"}, {"gender": "M", "name": "Bob"}]
    assert groupby("gender", seq) == {"F": [seq[0]], "M": [seq[1]]}


def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_with_key():
    out = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    assert out == [2, 1, 3, 3]


def test_interleave_two_sequences():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]


def test_unique_preserves_first_occurrence():
    assert tuple(unique([1, 2, 1, 3, 2, 4])) == (1, 2, 3, 4)


def test_unique_with_key():
    assert tuple(unique(["cat", "mouse", "dog", "hen"], key=len)) == ("cat", "mouse")


def test_isiterable_recognizes_strings_and_numbers():
    assert isiterable("abc") is True
    assert isiterable([1, 2, 3]) is True
    assert isiterable(5) is False


def test_isdistinct_sequence_true_false():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False


def test_isdistinct_iterator_consumes_and_detects_duplicates():
    it = iter([1, 2, 3, 2])
    assert isdistinct(it) is False


def test_take_drop_take_nth():
    assert list(take(2, [10, 20, 30])) == [10, 20]
    assert list(drop(2, [10, 20, 30])) == [30]
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_first_second_nth_last():
    assert first("ABC") == "A"
    assert second("ABC") == "B"
    assert nth(1, "ABC") == "B"
    assert last("ABC") == "C"


def test_rest_alias():
    assert list(rest([1, 2, 3])) == [2, 3]


def test_get_single_and_multiple():
    assert get(1, "ABC") == "B"
    assert get([1, 2], "ABC") == ("B", "C")


def test_get_with_default_and_missing():
    assert get("x", {"x": 1}) == 1
    assert get("y", {"x": 1}, default=None) is None


def test_get_empty_list_ind_returns_empty_tuple():
    assert get([], "ABC") == ()


def test_concat_and_concatv_and_mapcat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concatv([], ["a"], ["b", "c"])) == ["a", "b", "c"]
    assert list(mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c"]])) == ["A", "B", "C"]


def test_cons_and_interpose():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(interpose("x", [1, 2, 3])) == [1, "x", 2, "x", 3]


def test_frequencies_counts_items():
    assert frequencies(["cat", "cat", "ox", "pig", "pig", "cat"]) == {"cat": 3, "ox": 1, "pig": 2}


def test_reduceby_with_initial_constant():
    data = [1, 2, 3, 4, 5]
    out = reduceby(lambda x: x % 2 == 0, add, data, 0)
    assert out == {False: 9, True: 6}


def test_reduceby_without_initial_uses_first_item():
    data = [1, 2, 3, 4]
    out = reduceby(lambda x: x % 2, add, data)
    assert out == {1: 4, 0: 6}


def test_iterate_generates_infinite_sequence():
    it = iterate(lambda x: x + 1, 0)
    assert [next(it) for _ in range(4)] == [0, 1, 2, 3]


def test_sliding_window_basic():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_partition_without_pad_drops_remainder():
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad_uses_fillvalue():
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_all_keeps_remainder():
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_partition_all_on_empty_sequence():
    assert list(partition_all(3, [])) == []


def test_count_uses_len_when_available():
    assert count([1, 2, 3]) == 3


def test_count_consumes_lazy_iterator():
    assert count(i for i in range(5)) == 5


def test_pluck_single_key_and_list_key():
    data = [{"id": 1, "name": "Cheese"}, {"id": 2, "name": "Pies"}]
    assert list(pluck("name", data)) == ["Cheese", "Pies"]
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_join_inner_join():
    friends = [("Alice", "Edith"), ("Alice", "Zhao"), ("Edith", "Alice")]
    cities = [("Alice", "NYC"), ("Edith", "Paris"), ("Zhao", "Shanghai")]
    result = list(join(second, friends, first, cities))
    assert (("Alice", "Edith"), ("Edith", "Paris")) in result
    assert (("Alice", "Zhao"), ("Zhao", "Shanghai")) in result
    assert (("Edith", "Alice"), ("Alice", "NYC")) in result


def test_join_full_outer_join():
    out = list(join(lambda x: x, [1, 2, 3], lambda x: x, [2, 3, 4], left_default=None, right_default=None))
    assert set(out) == {(2, 2), (3, 3), (None, 4), (1, None)}


def test_diff_basic_and_default():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]


def test_diff_with_key():
    assert list(diff(["apples", "bananas"], ["Apples", "Oranges"], key=str.lower)) == [("bananas", "Oranges")]


def test_topk_basic_and_key():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ["Alice", "Bob", "Charlie", "Dan"], key=len) == ("Charlie", "Alice")


def test_peek_returns_first_and_replayable_iterator():
    first_item, seq = peek([0, 1, 2])
    assert first_item == 0
    assert list(seq) == [0, 1, 2]


def test_peekn_returns_multiple_and_replayable_iterator():
    first_two, seq = peekn(2, [0, 1, 2, 3])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3]


def test_random_sample_deterministic_with_seed():
    seq = list(range(20))
    sample1 = list(random_sample(0.3, seq, random_state=123))
    sample2 = list(random_sample(0.3, seq, random_state=123))
    assert sample1 == sample2


def test_random_sample_accepts_random_object():
    from random import Random

    seq = list(range(20))
    r1 = list(random_sample(0.25, seq, random_state=Random(99)))
    r2 = list(random_sample(0.25, seq, random_state=Random(99)))
    assert r1 == r2


def test_identity_and_apply():
    assert identity(3) == 3
    assert apply(add, 2, 3) == 5


def test_thread_first_and_thread_last():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert thread_first(1, inc, double) == 4
    assert thread_last(1, inc, double) == 4


def test_thread_first_and_last_with_tuples():
    def add2(x, y):
        return x + y

    def power(x, y):
        return x**y

    assert thread_first(1, (add2, 4), (power, 2)) == 25
    assert thread_last(1, (add2, 4), (power, 2)) == 32


def test_compose_and_compose_left_and_pipe():
    inc = lambda i: i + 1
    assert compose(str, inc)(3) == "4"
    assert compose_left(inc, str)(3) == "4"
    assert pipe(3, lambda x: x + 1, str) == "4"


def test_complement_inverts_predicate():
    iseven = lambda n: n % 2 == 0
    isodd = complement(iseven)
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt_multiple_functions():
    inc = lambda x: x + 1
    double = lambda x: x * 2
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_do_performs_side_effect_and_returns_input():
    seen = []

    def record(x):
        seen.append(x)

    assert do(record, 5) == 5
    assert seen == [5]


def test_flip_curried():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_excepts_catches_and_handles():
    exc = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert exc(1) == 0
    assert exc(3) == -1


def test_excepts_default_handler_returns_none():
    exc = excepts((IndexError, KeyError), lambda a: a[0])
    assert exc([]) is None
    assert exc([1]) == 1


def test_curry_basic_and_keywords():
    def mul2(x, y):
        return x * y

    curried = curry(mul2)
    assert curried(2)(10) == 20

    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5


def test_memoize_caches_results():
    calls = {"n": 0}

    @memoize
    def add2(x, y):
        calls["n"] += 1
        return x + y

    assert add2(1, 2) == 3
    assert add2(1, 2) == 3
    assert calls["n"] == 1


def test_memoize_custom_key_ignores_kwargs():
    calls = {"n": 0}

    @memoize(key=lambda args, kwargs: args)
    def add3(x, y, verbose=False):
        calls["n"] += 1
        return x + y

    assert add3(1, 2, verbose=True) == 3
    assert add3(1, 2, verbose=False) == 3
    assert calls["n"] == 1


def test_is_arity_and_signature_helpers():
    def f(x):
        return x

    def g(x, y=1):
        return x + y

    assert is_arity(1, f) is True
    assert is_arity(1, g) is False
    assert num_required_args(g) == 1
    assert has_varargs(f) is False
    assert has_keywords(g) is True
    assert is_valid_args(g, (1,), {}) is True
    assert is_valid_args(g, (), {}) is False
    assert is_partial_args(g, (1,), {}) is True
    assert is_partial_args(g, (1, 2, 3), {}) is False


def test_merge_and_merge_with():
    assert merge({1: "one"}, {2: "two"}) == {1: "one", 2: "two"}
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_valmap_keymap_itemmap():
    d = {"Alice": [1, 2], "Bob": [3]}
    assert valmap(sum, d) == {"Alice": 3, "Bob": 3}
    assert keymap(str.lower, d) == {"alice": [1, 2], "bob": [3]}
    assert itemmap(lambda kv: (kv[1], kv[0]), {"Alice": 10, "Bob": 20}) == {10: "Alice", 20: "Bob"}


def test_valfilter_keyfilter_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(lambda x: x % 2 == 0, d) == {1: 2, 3: 4}
    assert keyfilter(lambda x: x % 2 == 0, d) == {2: 3, 4: 5}
    assert itemfilter(lambda item: item[0] % 2 == 0 and item[1] < 4, d) == {2: 3}


def test_assoc_dissoc_and_nested_updates():
    assert assoc({"x": 1}, "x", 2) == {"x": 2}
    assert dissoc({"x": 1, "y": 2}, "y") == {"x": 1}
    purchase = {"name": "Alice", "order": {"costs": [0.5, 1.25]}}
    updated = assoc_in(purchase, ["order", "costs"], [0.25, 1.0])
    assert updated["order"]["costs"] == [0.25, 1.0]
    assert purchase["order"]["costs"] == [0.5, 1.25]


def test_update_in_existing_and_missing_paths():
    inc = lambda x: x + 1
    assert update_in({"a": 0}, ["a"], inc) == {"a": 1}
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: "bar"}}}


def test_get_in_success_and_default():
    transaction = {"name": "Alice", "purchase": {"items": ["Apple", "Orange"]}}
    assert get_in(["purchase", "items", 0], transaction) == "Apple"
    assert get_in(["purchase", "total"], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(["y"], {}, no_default=True)


def test_countby_and_partitionby():
    assert countby(len, ["cat", "mouse", "dog"]) == {3: 2, 5: 1}
    assert list(partitionby(lambda c: c == " ", "I have")) == [("I",), (" ",), ("h", "a", "v", "e")]


def test_equalityhashkey_and_unzip():
    k1 = EqualityHashKey(None, [1, 2])
    k2 = EqualityHashKey(None, [1, 2])
    assert k1 == k2
    assert str(k1) == "=[1, 2]="
    a, b = unzip([("a", 1), ("b", 2)])
    assert list(a) == ["a", "b"]
    assert list(b) == [1, 2]


def test_fold_reduces_in_chunks():
    assert fold(add, [1, 2, 3, 4], chunksize=2, map=map) == 10
    assert fold(mul, [1, 2, 3, 4], default=1, chunksize=2, map=map) == 24
