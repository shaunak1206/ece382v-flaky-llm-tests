from collections import OrderedDict

import pytest

from toolz.dicttoolz import (
    assoc, assoc_in, dissoc, get_in, itemfilter, itemmap, keyfilter, keymap,
    merge, merge_with, update_in, valfilter, valmap,
)
from toolz.recipes import countby, partitionby


def inc(x):
    return x + 1


def iseven(x):
    return x % 2 == 0


def test_merge_and_merge_with():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}
    # later dictionaries take precedence
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    assert merge() == {}
    assert merge([{1: 'a'}, {2: 'b'}]) == {1: 'a', 2: 'b'}
    result = merge({'a': 1}, {'b': 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    with pytest.raises(TypeError):
        merge({1: 2}, bogus=True)

    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30}) == \
        {1: [1], 2: [2, 20], 3: [30]}
    assert merge_with(sum, [{1: 1}, {1: 5}]) == {1: 6}
    with pytest.raises(TypeError):
        merge_with(sum, {1: 1}, bogus=True)


def test_valmap_keymap_itemmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {'Alice': 65, 'Bob': 45}
    assert keymap(str.lower, bills) == {'alice': [20, 15, 30],
                                        'bob': [10, 35]}
    assert itemmap(lambda kv: (kv[1], kv[0]), {"Alice": 10, "Bob": 20}) == \
        {10: "Alice", 20: "Bob"}
    assert valmap(inc, {}) == {}
    assert isinstance(valmap(inc, {'a': 1}, factory=OrderedDict), OrderedDict)
    assert isinstance(keymap(str, {1: 1}, factory=OrderedDict), OrderedDict)
    # the input is not mutated
    assert bills["Bob"] == [10, 35]


def test_valfilter_keyfilter_itemfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert valfilter(iseven, d) == {1: 2, 3: 4}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}
    assert valfilter(lambda v: False, d) == {}
    assert isinstance(keyfilter(iseven, d, factory=OrderedDict), OrderedDict)
    assert d == {1: 2, 2: 3, 3: 4, 4: 5}


def test_assoc_and_dissoc_do_not_mutate():
    original = {'x': 1}
    assert assoc(original, 'x', 2) == {'x': 2}
    assert assoc(original, 'y', 3) == {'x': 1, 'y': 3}
    assert original == {'x': 1}

    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}
    assert dissoc({'x': 1}, 'y') == {'x': 1}
    # few keys removed from a big dict, and many keys removed from a small one
    big = {i: i for i in range(10)}
    assert dissoc(big, 0) == {i: i for i in range(1, 10)}
    assert dissoc(big, 0, 1, 2, 3, 4, 5, 6) == {7: 7, 8: 8, 9: 9}
    assert len(big) == 10
    assert isinstance(dissoc(big, 0, factory=OrderedDict), OrderedDict)
    with pytest.raises(TypeError):
        dissoc(big, 0, bogus=1)


def test_update_in_and_assoc_in():
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == \
        {1: 'foo', 2: {3: {4: 1}}}

    nested = {'name': 'Alice', 'purchase': {'costs': [0.5, 1.25]}}
    updated = update_in(nested, ['purchase', 'costs'], sum)
    assert updated == {'name': 'Alice', 'purchase': {'costs': 1.75}}
    assert nested['purchase']['costs'] == [0.5, 1.25]

    assert assoc_in({'a': {'b': 1}}, ['a', 'c'], 2) == {'a': {'b': 1, 'c': 2}}
    assert assoc_in({}, ['x', 'y'], 5) == {'x': {'y': 5}}
    assert assoc_in({'a': 1}, ['a'], 9) == {'a': 9}


def test_get_in():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in([], transaction) == transaction
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)
    with pytest.raises(IndexError):
        get_in([5], [1], no_default=True)


def test_countby():
    assert countby(len, ['cat', 'mouse', 'dog']) == {3: 2, 5: 1}
    assert countby(iseven, [1, 2, 3]) == {True: 1, False: 2}
    assert countby(0, [('a', 1), ('b', 2), ('a', 3)]) == {'a': 2, 'b': 1}
    assert countby(len, []) == {}


def test_partitionby():
    is_space = lambda c: c == " "  # noqa: E731
    assert list(partitionby(is_space, "I have space")) == \
        [('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',),
         ('s', 'p', 'a', 'c', 'e')]
    is_large = lambda x: x > 10  # noqa: E731
    assert list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5])) == \
        [(1, 2, 1), (99, 88, 33, 99), (-1, 5)]
    assert list(partitionby(iseven, [])) == []
    assert list(partitionby(iseven, [2, 4, 6])) == [(2, 4, 6)]
