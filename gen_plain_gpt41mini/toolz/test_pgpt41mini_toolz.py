import pytest
import operator
import collections
import itertools
import random
from types import MethodType

from toolz import itertoolz, functoolz, dicttoolz, recipes
from toolz.sandbox import core, parallel


# ----------------------------------------
# Tests for toolz.itertoolz functions
# ----------------------------------------

def test_remove_basic():
    is_even = lambda x: x % 2 == 0
    result = list(itertoolz.remove(is_even, [1, 2, 3, 4]))
    assert result == [1, 3]


@pytest.mark.parametrize("binop,seq,initial,expected", [
    (operator.add, [1, 2, 3, 4], itertoolz.no_default, [1, 3, 6, 10]),
    (operator.mul, [1, 2, 3, 4], itertoolz.no_default, [1, 2, 6, 24]),
    (operator.add, [1, 2, 3], -1, [-1, 0, 2, 5]),
    (operator.add, [], 1, [1]),
])
def test_accumulate(binop, seq, initial, expected):
    result = list(itertoolz.accumulate(binop, seq, initial))
    assert result == expected


def test_accumulate_empty_no_initial():
    result = list(itertoolz.accumulate(operator.add, []))
    assert result == []


def test_groupby_callable_key_and_attribute_key():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    grouped_len = itertoolz.groupby(len, names)
    assert grouped_len[3] == ['Bob', 'Dan']
    assert 'Charlie' in grouped_len[7]

    dicts = [{'name': 'Alice', 'gender': 'F'},
             {'name': 'Bob', 'gender': 'M'},
             {'name': 'Charlie', 'gender': 'M'}]
    grouped_gender = itertoolz.groupby('gender', dicts)
    assert grouped_gender['F'] == [{'name': 'Alice', 'gender': 'F'}]
    assert len(grouped_gender['M']) == 2


def test_merge_sorted_basic_and_key():
    seqs = [[1, 3, 5], [2, 4, 6]]
    result = list(itertoolz.merge_sorted(*seqs))
    assert result == sorted([x for seq in seqs for x in seq])

    result2 = list(itertoolz.merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
    # Because key groups items differently, result may not be sorted numerically here:
    assert sorted(result2) == [1, 2, 3, 3]


def test_merge_sorted_empty_and_single_sequence():
    assert list(itertoolz.merge_sorted()) == []
    seq = [1, 2, 3]
    assert list(itertoolz.merge_sorted(seq)) == seq


def test_interleave_finite_sequences():
    res = list(itertoolz.interleave([[1, 2], [3, 4]]))
    assert res == [1, 3, 2, 4]

    res2 = ''.join(itertoolz.interleave(('ABC', 'XY')))
    assert res2 == 'AXBYC'


def test_unique_no_key_and_with_key():
    seq = [1, 2, 1, 3]
    uniq = tuple(itertoolz.unique(seq))
    assert set(uniq) == {1, 2, 3}

    words = ['cat', 'mouse', 'dog', 'hen']
    uniq_len = tuple(itertoolz.unique(words, key=len))
    assert 'cat' in uniq_len
    assert 'mouse' in uniq_len
    assert 'hen' not in uniq_len or 'dog' not in uniq_len  # only one of len=3


@pytest.mark.parametrize("value,expected", [
    ([1, 2, 3], True),
    ([1, 2, 1], False),
    ("Hello", False),
    ("World", True),
])
def test_isdistinct(value, expected):
    assert itertoolz.isdistinct(value) is expected


def test_isiterable_true_and_false():
    assert itertoolz.isiterable([1, 2, 3])
    assert itertoolz.isiterable('abc')
    assert not itertoolz.isiterable(5)
    assert itertoolz.isiterable({'a':1})


def test_take_and_drop_and_take_nth():
    lst = list(range(10))
    t = list(itertoolz.take(3, lst))
    assert t == [0,1,2]

    d = list(itertoolz.drop(8, lst))
    assert d == [8,9]

    tn = list(itertoolz.take_nth(3, lst))
    assert tn == [0, 3, 6, 9]


def test_first_second_nth_last():
    seq = 'ABCDE'
    assert itertoolz.first(seq) == 'A'
    assert itertoolz.second(seq) == 'B'
    assert itertoolz.nth(2, seq) == 'C'
    assert itertoolz.last(seq) == 'E'


def test_nth_with_list_and_iterable():
    seq = [10, 20, 30]
    assert itertoolz.nth(1, seq) == 20
    gen = (x for x in seq)
    assert itertoolz.nth(2, gen) == 30


def test_rest_alias_for_drop():
    data = [1, 2, 3]
    rest_res = list(itertoolz.rest(data))
    drop_res = list(itertoolz.drop(1, data))
    assert rest_res == drop_res == [2, 3]


def test_get_with_single_and_multiple_indices_and_default():
    seq = 'ABCDE'
    assert itertoolz.get(1, seq) == 'B'
    assert itertoolz.get([1, 2], seq) == ('B', 'C')
    d = {'a': 1, 'b': 2}
    assert itertoolz.get('a', d) == 1
    assert itertoolz.get(['a', 'b'], d) == (1, 2)
    assert itertoolz.get(['a', 'c'], d, default=0) == (1, 0)
    # Default not specified raises
    with pytest.raises(KeyError):
        itertoolz.get('c', d)


def test_concat_and_concatv_and_mapcat():
    parts = [[], [1], [2, 3]]
    assert list(itertoolz.concat(parts)) == [1, 2, 3]

    assert list(itertoolz.concatv([], ['a'], ['b', 'c'])) == ['a', 'b', 'c']

    seqs = [['a', 'b'], ['c', 'd', 'e']]
    mapped = itertoolz.mapcat(lambda s: [c.upper() for c in s], seqs)
    assert tuple(mapped) == ('A', 'B', 'C', 'D', 'E')


def test_cons_adds_element_first():
    res = list(itertoolz.cons(1, [2, 3]))
    assert res == [1, 2, 3]


def test_interpose_basic_and_inposed_behavior():
    seq = [1, 2, 3]
    res = list(itertoolz.interpose("a", seq))
    assert res == [1, "a", 2, "a", 3]


def test_frequencies_basic():
    seq = ['cat', 'cat', 'ox', 'pig', 'pig', 'cat']
    freqs = itertoolz.frequencies(seq)
    assert freqs == {'cat': 3, 'ox': 1, 'pig': 2}


def test_reduceby_basic_add_and_mul():
    data = [1, 2, 3, 4, 5]
    is_even = lambda x: x % 2 == 0
    addres = itertoolz.reduceby(is_even, operator.add, data)
    mulres = itertoolz.reduceby(is_even, operator.mul, data)

    # keys are booleans
    assert addres[True] == 6 and addres[False] == 9
    assert mulres[False] == 15 and mulres[True] == 8


def test_reduceby_with_init_callable_and_non_callable():
    data = [1, 2, 3, 4, 5]
    def set_add(s, i):
        s.add(i)
        return s

    res = itertoolz.reduceby(lambda x: x % 2 == 0, set_add, data, set)
    assert isinstance(res[True], set)
    assert isinstance(res[False], set)
    assert 2 in res[True]
    assert 3 in res[False]

    res2 = itertoolz.reduceby(lambda x: x % 2 == 0, operator.add, data, 0)
    assert res2[True] == 6 and res2[False] == 9


def test_iterate_basic_functionality():
    inc = lambda x: x + 1
    counter = itertoolz.iterate(inc, 0)
    assert next(counter) == 0
    assert next(counter) == 1
    assert next(counter) == 2


def test_sliding_window_basic_and_mean():
    data = [1, 2, 3, 4]
    windows = list(itertoolz.sliding_window(2, data))
    assert windows == [(1, 2), (2, 3), (3, 4)]

    mean = lambda seq: float(sum(seq)) / len(seq)
    means = list(map(mean, itertoolz.sliding_window(2, data)))
    assert means == [1.5, 2.5, 3.5]


def test_partition_and_partition_all_and_pad_behavior():
    seq = [1, 2, 3, 4, 5]
    p = list(itertoolz.partition(2, seq))
    assert p == [(1, 2), (3, 4)]

    p_pad = list(itertoolz.partition(2, seq, pad=None))
    assert p_pad[-1] == (5, None)

    pa = list(itertoolz.partition_all(2, seq))
    assert pa[-1] == (5,)


def test_count_len_and_iterable():
    l = [1, 2, 3]
    g = (x for x in l)
    assert itertoolz.count(l) == 3
    assert itertoolz.count(g) == 3


def test_pluck_single_and_multiple_indices_with_default():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    names = list(itertoolz.pluck('name', data))
    assert names == ['Cheese', 'Pies']

    nested = [[1, 2, 3], [4, 5, 6]]
    res = list(itertoolz.pluck([0, 1], nested))
    assert res == [(1, 2), (4, 5)]

    # With default value for missing keys
    data2 = [{'a': 1}, {'a': 2, 'b': 4}]
    res2 = list(itertoolz.pluck(['a', 'b'], data2, default=0))
    assert res2 == [(1, 0), (2, 4)]


def test_getter_single_and_multiple_indices():
    f1 = itertoolz.getter(0)
    assert f1((9, 10)) == 9

    f2 = itertoolz.getter([0])
    assert f2((9, 10)) == (9,)

    f3 = itertoolz.getter([0, 1])
    assert f3((9, 10)) == (9, 10)

    f4 = itertoolz.getter([])
    assert f4((9, 10)) == ()


def test_join_inner_right_left_full_joins():
    friends = [('Alice', 'Edith'),
               ('Alice', 'Zhao'),
               ('Edith', 'Alice')]

    cities = [('Alice', 'NYC'),
              ('Alice', 'Chicago'),
              ('Dan', 'Sydney'),
              ('Edith', 'Paris')]

    # Inner join
    result = set(itertoolz.join(lambda x: x[1], friends, lambda x: x[0], cities))
    # Alice appears with Edith and Zhao friends, Edith with Alice city, Dan only in cities so no output here
    assert any('Alice' in l for l, r in result)

    # Right join (left_default)
    result = list(itertoolz.join(lambda x: x[1], friends, lambda x: x[0], cities, left_default=None))
    assert any(left is None for left, _ in result)

    # Left join (right_default)
    result = list(itertoolz.join(lambda x: x[1], friends, lambda x: x[0], cities, right_default=None))
    assert any(right is None for _, right in result)

    # Full join both defaults
    result = list(itertoolz.join(lambda x: x[1], friends, lambda x: x[0], cities, left_default=None, right_default=None))
    assert any(left is None or right is None for left, right in result)


def test_diff_basic_and_with_key_and_default():
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 10, 100]
    diff_res = list(itertoolz.diff(seq1, seq2))
    assert (3, 10) in diff_res

    diff_res2 = list(itertoolz.diff(seq1, seq2, default=None))
    assert (None, 100) in diff_res2

    diff_key = list(itertoolz.diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
    assert ('bananas', 'Oranges') in diff_key


def test_topk_without_and_with_key():
    data = [1, 100, 10, 1000]
    top2 = itertoolz.topk(2, data)
    assert top2 == (1000, 100)

    words = ['Alice', 'Bob', 'Charlie', 'Dan']
    top_longest = itertoolz.topk(2, words, key=len)
    assert top_longest == ('Charlie', 'Alice')


def test_peek_and_peekn_iterators():
    seq = [0, 1, 2, 3, 4]
    first_elem, new_seq = itertoolz.peek(seq)
    assert first_elem == 0
    assert list(new_seq) == seq

    first_two, new_seq2 = itertoolz.peekn(2, seq)
    assert first_two == (0, 1)
    assert list(new_seq2) == seq


def test_random_sample_probability_and_seed_consistency():
    seq = list(range(100))
    samples1 = list(itertoolz.random_sample(0.1, seq))
    samples2 = list(itertoolz.random_sample(0.1, seq))
    # Due to randomness, these should differ mostly
    assert not samples1 == samples2 or len(set(samples1).intersection(samples2)) < len(samples1)

    # With seed, results should be consistent
    seed_samples1 = list(itertoolz.random_sample(0.1, seq, random_state=2016))
    seed_samples2 = list(itertoolz.random_sample(0.1, seq, random_state=2016))
    assert seed_samples1 == seed_samples2


@pytest.mark.parametrize("x,value", [
    ([1, 2, 3], 6),
    ([], 0),
    ((i for i in range(4)), 6),
])
def test_count_function(x, value):
    assert itertoolz.count(x) == value


# ----------------------------------------
# Tests for toolz.functoolz functions
# ----------------------------------------

def test_identity_function():
    assert functoolz.identity(5) == 5

def test_apply_function():
    def add(x, y): return x + y
    assert functoolz.apply(add, 2, 3) == 5
    with pytest.raises(TypeError):
        functoolz.apply()

def test_thread_first_basic_and_complex():
    def add(x, y): return x + y
    def pow(x, y): return x ** y

    result = functoolz.thread_first(1, add, (pow, 2))
    assert result == 25

def test_thread_last_basic_and_complex():
    def add(x, y): return x + y
    def pow(x, y): return x ** y
    result = functoolz.thread_last(1, add, (pow, 2))
    assert result == 32

def test_instanceproperty_behavior():
    class C:
        @functoolz.instanceproperty(classval="Class doc")
        def __doc__(self):
            return "Instance doc"

        @functoolz.instanceproperty
        def val(self):
            return 42

    assert C.__doc__ == "Class doc"
    assert getattr(C, 'val', None) is None
    obj = C()
    assert obj.__doc__ == "Instance doc"
    assert obj.val == 42

def test_curry_basic_and_partial_application():
    @functoolz.curry
    def mul(x, y):
        return x * y
    double = mul(2)
    assert double(5) == 10
    # Using curry as decorator with kwarg
    @functoolz.curry
    def f(x, y, a=10):
        return a * (x + y)
    add_a1 = f(a=1)
    assert add_a1(2, 3) == 5

def test_memoize_basic_and_cache_and_key():
    calls = []
    @functoolz.memoize
    def f(x):
        calls.append(x)
        return x * 2
    # Call twice with same arg
    r1 = f(2)
    r2 = f(2)
    assert r1 == r2 == 4
    assert len(calls) == 1
    # Using cache argument
    cache = {(3, ): 9}
    @functoolz.memoize(cache=cache)
    def g(x):
        return x**2
    assert g(3) == 9

def test_compose_and_compose_left_and_pipe_work_as_expected():
    inc = lambda x: x + 1
    to_str = str

    comp = functoolz.compose(to_str, inc)
    assert comp(3) == '4'

    comp_left = functoolz.compose_left(inc, to_str)
    assert comp_left(3) == '4'

    assert functoolz.pipe(3, inc, to_str) == '4'

def test_complement_behavior():
    iseven = lambda x: x % 2 == 0
    isodd = functoolz.complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False

def test_juxt_function_works_with_funcs_and_list_of_funcs():
    inc = lambda x: x + 1
    dbl = lambda x: x * 2
    j = functoolz.juxt(inc, dbl)
    j2 = functoolz.juxt([inc, dbl])
    assert j(10) == (11, 20)
    assert j2(10) == (11, 20)

def test_do_runs_func_and_returns_x():
    log = []
    def logger(x):
        log.append(x)
    inc = lambda x: x + 1
    inc_with_log = functoolz.compose(inc, lambda x: functoolz.do(logger, x))
    res = inc_with_log(5)
    assert res == 6
    assert 5 in log

def test_flip_basic_behavior():
    def div(a, b): return a // b
    assert functoolz.flip(div, 2, 6) == 3
    div_by_two = functoolz.flip(div, 2)
    assert div_by_two(4) == 2
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(functoolz.flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]

def test_excepts_catch_exceptions_and_call_handler():
    def f(a):
        if a == 0:
            raise ValueError
        return 10 / a
    handler = lambda e: -1
    wrapped = functoolz.excepts(ValueError, f, handler)
    assert wrapped(1) == 10
    assert wrapped(0) == -1

def test_compose_repr_eq_hash():
    inc = lambda x: x + 1
    dbl = lambda x: x * 2
    c1 = functoolz.compose(inc, dbl)
    c2 = functoolz.compose(inc, dbl)
    assert repr(c1)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1(3) == inc(dbl(3))
    # Compose with no funcs returns identity
    assert functoolz.compose()(4) == 4
    # Compose with single func returns func
    assert functoolz.compose(inc)(4) == 5

def test_curry_eq_and_hash():
    @functoolz.curry
    def f(x, y): return x + y
    c1 = f(2)
    c2 = f(2)
    assert c1 == c2
    assert hash(c1) == hash(c2)


# ----------------------------------------
# Tests for toolz.dicttoolz functions
# ----------------------------------------

def test_merge_and_merge_with_basic():
    d1 = {1: 'one'}
    d2 = {2: 'two'}
    merged = dicttoolz.merge(d1, d2)
    assert merged == {1: 'one', 2: 'two'}

    d1 = {1: 2, 3: 4}
    d2 = {3: 3, 4: 4}
    merged2 = dicttoolz.merge(d1, d2)
    assert merged2[3] == 3

    merged_with_sum = dicttoolz.merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    assert merged_with_sum == {1: 11, 2: 22}

def test_valmap_keymap_itemmap():
    d = {"A": 1, "B": 2}
    valmapped = dicttoolz.valmap(lambda x: x*2, d)
    assert valmapped == {"A": 2, "B": 4}

    keymapped = dicttoolz.keymap(str.lower, d)
    assert "a" in keymapped and "b" in keymapped

    itemmapped = dicttoolz.itemmap(lambda kv: (kv[1], kv[0]), d)
    assert itemmapped == {1: "A", 2: "B"}

def test_valfilter_keyfilter_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    is_even = lambda x: x % 2 == 0
    val_filtered = dicttoolz.valfilter(is_even, d)
    assert val_filtered == {1: 2, 3: 4}

    key_filtered = dicttoolz.keyfilter(is_even, d)
    assert key_filtered == {2: 3, 4: 5}

    item_filtered = dicttoolz.itemfilter(lambda kv: kv[0] % 2 == 0 and kv[1] < 4, d)
    assert item_filtered == {2: 3}

def test_assoc_and_dissoc():
    d = {'x': 1}
    a = dicttoolz.assoc(d, 'x', 2)
    assert a['x'] == 2
    a2 = dicttoolz.assoc(d, 'y', 3)
    assert a2['y'] == 3

    d = {'x': 1, 'y': 2}
    diss = dicttoolz.dissoc(d, 'y')
    assert 'y' not in diss
    diss2 = dicttoolz.dissoc(d, 'y', 'x')
    assert diss2 == {}
    diss3 = dicttoolz.dissoc({'x': 1}, 'y')
    assert diss3 == {'x': 1}

def test_assoc_in_and_update_in_basic_and_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    updated = dicttoolz.assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert updated['order']['costs'] == [0.25, 1.00]

    increment = lambda x: x + 1
    result = dicttoolz.update_in({'a': 0}, ['a'], increment)
    assert result['a'] == 1

    nested_dict = {}
    res = dicttoolz.update_in(nested_dict, [1, 2, 3], str, default="bar")
    assert res[1][2][3] == "bar"

def test_get_in_basic_and_default_and_no_default():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert dicttoolz.get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert dicttoolz.get_in(['name'], transaction) == 'Alice'
    assert dicttoolz.get_in(['purchase', 'total'], transaction) is None
    assert dicttoolz.get_in(['purchase', 'total'], transaction, 0) == 0
    with pytest.raises(KeyError):
        dicttoolz.get_in(['y'], {}, no_default=True)


# ----------------------------------------
# Tests for toolz.recipes
# ----------------------------------------

def test_countby_basic_and_with_callable_key():
    data = ['cat', 'mouse', 'dog']
    counted = recipes.countby(len, data)
    assert counted == {3: 2, 5: 1}

    is_even = lambda x: x % 2 == 0
    data2 = [1, 2, 3]
    counted2 = recipes.countby(is_even, data2)
    assert counted2[True] == 1 and counted2[False] == 2


def test_partitionby_basic():
    text = "I have space"
    result = list(recipes.partitionby(lambda c: c == " ", text))
    assert all(isinstance(group, tuple) for group in result)
    assert any(' ' in group for group in result)

    data = [1, 2, 1, 99, 88, 33, 99, -1, 5]
    result2 = list(recipes.partitionby(lambda x: x > 10, data))
    assert all(isinstance(group, tuple) for group in result2)
    # Ensure partition switches on lambda output
    for group in result2:
        vals = set(x > 10 for x in group)
        assert len(vals) == 1


# ----------------------------------------
# Tests for toolz.sandbox.core
# ----------------------------------------

def test_equalityhashkey_hash_and_eq():
    EqualityHashDefault = core.EqualityHashKey(None)
    base_list = [[], (), [1], [1]]
    keys = set(map(core.EqualityHashKey(None), base_list))
    # Should contain unique items by equality
    assert len(keys) == 3

    key_with_len = core.EqualityHashKey(len, [1, 2])
    key_with_len2 = core.EqualityHashKey(len, ['a', 'b'])
    # Hash should be same because key returns same length=2
    assert hash(key_with_len) == hash(key_with_len2)
    # Equal returns False for different items
    assert key_with_len != key_with_len2 or key_with_len.item != key_with_len2.item

def test_equalityhashkey_str_repr():
    item = [42]
    key = core.EqualityHashKey(None, item)
    s = str(key)
    r = repr(key)
    assert s.startswith("=") and s.endswith("=")
    assert repr(key).startswith("=")

def test_unzip_basic_and_empty():
    zipped = [('a', 1), ('b', 2)]
    a, b = core.unzip(zipped)
    assert list(a) == ['a', 'b']
    assert list(b) == [1, 2]

    # Empty sequence returns empty tuple
    assert core.unzip([]) == ()


# ----------------------------------------
# Tests for toolz.sandbox.parallel
# ----------------------------------------

def test_reduce_function_and_fold_simple_sum():
    total = parallel.fold(operator.add, [1, 2, 3, 4], chunksize=2, map=map)
    assert total == 10

    total_with_default = parallel.fold(operator.add, [1, 2, 3, 4], default=0, chunksize=2, map=map)
    assert total_with_default == 10

def test_fold_recursive_behavior():
    # Folding with several chunks should recurse
    seq = list(range(100))
    total = parallel.fold(operator.add, seq, chunksize=10, map=map)
    assert total == sum(seq)


# -------------------------------
# Async tests if needed (pytest-asyncio)
# -------------------------------

import asyncio
import pytest

@pytest.mark.asyncio
async def test_async_iterate_usage():
    async def async_gen():
        for i in range(3):
            yield i

    it = itertoolz.iterate(lambda x: x + 1, 0)
    first_three = [next(it) for _ in range(3)]
    assert first_three == [0, 1, 2]

# -------------------------------
# Additional sanity tests
# -------------------------------

def test_cons_and_interpose_iterators_do_not_exhaust():
    # Test cons returns iterator with correct elements
    seq = iter([2, 3])
    new = itertoolz.cons(1, seq)
    assert list(new) == [1, 2, 3]

    # Test interpose returns iterator that interposes correctly
    seq = [1, 2, 3]
    interp = itertoolz.interpose(0, seq)
    # iterator, not exhausted by "next" call inside function
    assert list(interp) == [1, 0, 2, 0, 3]

def test_peek_raises_stopiteration_on_empty():
    with pytest.raises(StopIteration):
        itertoolz.peek([])

def test_peekn_works_for_empty_seq():
    res, seq = itertoolz.peekn(0, [])
    assert res == ()
    assert list(seq) == []

def test_diff_raises_with_one_sequence():
    with pytest.raises(TypeError):
        list(itertoolz.diff([1,2,3]))

def test_groupby_handles_empty_sequence():
    result = itertoolz.groupby(len, [])
    assert result == {}

def test_partition_and_partition_all_empty():
    assert list(itertoolz.partition(2, [])) == []
    assert list(itertoolz.partition_all(2, [])) == []

def test_topk_empty_sequence():
    assert itertoolz.topk(2, []) == ()

def test_frequencies_empty_sequence():
    assert dicttoolz.frequencies([]) == {}

def test_valfilter_empty_dict():
    d = {}
    res = dicttoolz.valfilter(lambda x: True, d)
    assert res == {}

def test_keyfilter_empty_dict():
    d = {}
    res = dicttoolz.keyfilter(lambda x: True, d)
    assert res == {}

def test_itemfilter_empty_dict():
    d = {}
    res = dicttoolz.itemfilter(lambda kv: True, d)
    assert res == {}

def test_assoc_in_creates_nested_dict():
    d = {}
    result = dicttoolz.assoc_in(d, ['a', 'b'], 1)
    assert 'a' in result and 'b' in result['a'] and result['a']['b'] == 1


# Run the test if this file run as main
if __name__ == "__main__":
    pytest.main([__file__])
