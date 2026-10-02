import itertools

import pytest

import toolz
from toolz import countby, curry, partitionby, pipe, unique
from toolz.curried import (
    assoc,
    drop,
    get,
    get_in,
    groupby,
    keyfilter,
    merge,
    merge_with,
    pluck,
    reduceby,
    take,
    valfilter,
    valmap,
)
from toolz.sandbox import EqualityHashKey, unzip


def iseven(x):
    return x % 2 == 0


# ---------------------------------------------------------------- recipes


def test_countby():
    assert countby(len, ['cat', 'mouse', 'dog']) == {3: 2, 5: 1}
    assert countby(iseven, [1, 2, 3]) == {True: 1, False: 2}
    assert countby(0, ['ab', 'ac', 'b']) == {'a': 2, 'b': 1}
    assert countby(len, []) == {}


def test_partitionby():
    is_space = lambda c: c == " "  # noqa: E731
    assert list(partitionby(is_space, "I have space")) == [
        ('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')]
    is_large = lambda x: x > 10  # noqa: E731
    assert list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5])) == [
        (1, 2, 1), (99, 88, 33, 99), (-1, 5)]
    assert list(partitionby(is_large, [])) == []


# ---------------------------------------------------------------- curried


def test_curried_get_in_map():
    data = [(1, 2), (11, 22), (111, 222)]
    assert get(0, ('a', 'b')) == 'a'
    assert list(map(get(0), data)) == [1, 11, 111]


def test_curried_objects_groupby_valmap():
    assert isinstance(groupby, curry)
    assert isinstance(valmap, curry)
    assert isinstance(take, curry)
    names = ['Alice', 'Bob', 'Charlie', 'Dan']
    by_len = groupby(len)
    assert by_len(names) == {5: ['Alice'], 3: ['Bob', 'Dan'], 7: ['Charlie']}
    assert valmap(sum)({'a': [1, 2], 'b': [3]}) == {'a': 3, 'b': 3}


def test_curried_take_drop_pluck():
    assert list(take(2)([10, 20, 30])) == [10, 20]
    assert list(drop(2)([10, 20, 30])) == [30]
    assert list(pluck('k')([{'k': 1}, {'k': 2}])) == [1, 2]


def test_curried_merge_and_merge_with():
    assert merge({1: 2}, {3: 4}) == {1: 2, 3: 4}
    assert merge_with(sum)({1: 1, 2: 2}, {1: 10}) == {1: 11, 2: 2}
    assert merge_with(sum, {1: 1}, {1: 2}) == {1: 3}


def test_curried_dict_helpers_and_reduceby():
    assert assoc({'x': 1})('y', 2) == {'x': 1, 'y': 2}
    assert get_in(['a', 'b'])({'a': {'b': 5}}) == 5
    assert valfilter(iseven)({1: 2, 2: 3}) == {1: 2}
    assert keyfilter(iseven)({1: 2, 2: 3}) == {2: 3}
    result = reduceby(iseven)(lambda a, b: a + b)([1, 2, 3, 4, 5])
    assert result == {False: 9, True: 6}


def test_curried_pipeline_with_builtins():
    from toolz.curried import filter as cfilter
    from toolz.curried import map as cmap

    result = pipe(
        [1, 2, 3, 4],
        cfilter(iseven),
        cmap(lambda x: x * 10),
        list,
    )
    assert result == [20, 40]


# ---------------------------------------------------------------- sandbox


def test_unzip_columns_and_empty():
    a, b = unzip([('a', 1), ('b', 2)])
    assert list(a) == ['a', 'b']
    assert list(b) == [1, 2]
    xs, ys, zs = unzip([(1, 'a', True), (2, 'b', False)])
    assert list(xs) == [1, 2]
    assert list(ys) == ['a', 'b']
    assert list(zs) == [True, False]
    assert unzip([]) == ()


def test_unzip_infinite_sequence_and_inverts_zip():
    a, b = unzip(zip(itertools.count(), itertools.count(10)))
    assert list(toolz.take(3, a)) == [0, 1, 2]
    assert list(toolz.take(3, b)) == [10, 11, 12]
    left, right = [1, 2, 3], ['x', 'y', 'z']
    c, d = unzip(zip(left, right))
    assert list(c) == left
    assert list(d) == right


def test_equalityhashkey_makes_lists_hashable():
    EqualityHashDefault = curry(EqualityHashKey, None)
    keys = set(map(EqualityHashDefault, [[], (), [1], [1]]))
    assert len(keys) == 3
    assert EqualityHashDefault([1]) in keys
    assert EqualityHashDefault([2]) not in keys


def test_equalityhashkey_eq_and_ne():
    a = EqualityHashKey(None, [1, 2])
    b = EqualityHashKey(None, [1, 2])
    c = EqualityHashKey(None, [3])
    assert a == b
    assert a != c
    assert not (a != b)
    assert a != [1, 2]


def test_equalityhashkey_hash_with_key_function():
    h1 = EqualityHashKey(len, [1, 2])
    h2 = EqualityHashKey(len, [7, 8])
    h3 = EqualityHashKey(len, [1])
    assert hash(h1) == hash(h2) == hash(2)
    assert hash(h1) != hash(h3)
    assert h1 != h2
    assert len({h1, h2, h3}) == 3


def test_equalityhashkey_with_unique_and_str_repr():
    EqualityHashDefault = curry(EqualityHashKey, None)
    vals = [[], [], (), [1], [1], [2], {}, {}, {}]
    assert list(unique(vals, key=EqualityHashDefault)) == [
        [], (), [1], [2], {}]
    k = EqualityHashKey(None, [1])
    assert str(k) == '=[1]='
    assert repr(k) == '=[1]='


def test_equalityhashkey_index_key():
    k1 = EqualityHashKey(0, [5, 6])
    k2 = EqualityHashKey(0, [5, 7])
    assert hash(k1) == hash(k2) == hash(5)
    assert k1 != k2
    with pytest.raises(TypeError):
        hash([5, 6])
