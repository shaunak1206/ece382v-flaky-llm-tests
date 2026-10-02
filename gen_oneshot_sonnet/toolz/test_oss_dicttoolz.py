from collections import OrderedDict

import pytest

from toolz import (
    assoc,
    assoc_in,
    dissoc,
    get_in,
    itemfilter,
    itemmap,
    keyfilter,
    keymap,
    merge,
    merge_with,
    update_in,
    valfilter,
    valmap,
)


def iseven(x):
    return x % 2 == 0


def inc(x):
    return x + 1


def test_merge_basic_cases():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    assert merge([{1: 2}, {3: 4}, {1: 5}]) == {1: 5, 3: 4}
    assert merge() == {}


def test_merge_does_not_mutate_inputs():
    a = {'x': 1}
    b = {'y': 2}
    result = merge(a, b)
    result['z'] = 3
    assert a == {'x': 1}
    assert b == {'y': 2}


def test_merge_factory_and_unknown_keyword():
    result = merge({1: 'a'}, {2: 'b'}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [(1, 'a'), (2, 'b')]
    with pytest.raises(TypeError):
        merge({1: 2}, {3: 4}, bogus=True)


def test_merge_with_combines_values():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}
    assert merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: [1], 2: [2, 20], 3: [30]}


def test_merge_with_sequence_of_dicts_and_factory():
    result = merge_with(max, [{'a': 1}, {'a': 5}, {'a': 3}],
                        factory=OrderedDict)
    assert result == {'a': 5}
    assert isinstance(result, OrderedDict)


def test_valmap_keymap_itemmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {'Alice': 65, 'Bob': 45}
    assert keymap(str.lower, bills) == {'alice': [20, 15, 30], 'bob': [10, 35]}
    accountids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), accountids) == {
        10: "Alice", 20: "Bob"}


def test_map_functions_respect_factory_and_empty():
    assert valmap(inc, {}) == {}
    result = valmap(inc, {'a': 1, 'b': 2}, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [('a', 2), ('b', 3)]


def test_valfilter_and_keyfilter():
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}


def test_itemfilter_and_filter_factory():
    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert itemfilter(isvalid, d) == {2: 3}

    d = {'a': 1, 'b': 2, 'c': 3}
    result = valfilter(lambda v: v > 1, d, factory=OrderedDict)
    assert isinstance(result, OrderedDict)
    assert d == {'a': 1, 'b': 2, 'c': 3}
    assert list(keyfilter(lambda k: k != 'b', d)) == ['a', 'c']


def test_assoc_adds_and_replaces_without_mutation():
    d = {'x': 1}
    assert assoc(d, 'x', 2) == {'x': 2}
    assert assoc(d, 'y', 3) == {'x': 1, 'y': 3}
    assert d == {'x': 1}


def test_dissoc_removes_keys():
    d = {'x': 1, 'y': 2}
    assert dissoc(d, 'y') == {'x': 1}
    assert dissoc(d, 'y', 'x') == {}
    assert dissoc({'x': 1}, 'y') == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_dissoc_many_keys_and_bad_keyword():
    d = {i: i * i for i in range(10)}
    result = dissoc(d, 0, 1, 2, 3, 4, 5, 6)
    assert result == {7: 49, 8: 64, 9: 81}
    assert len(d) == 10
    with pytest.raises(TypeError):
        dissoc({1: 2}, 1, nope=1)


def test_assoc_in():
    purchase = {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
        'credit card': '5555-1234-1234-1234',
    }
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
        'credit card': '5555-1234-1234-1234',
    }
    assert purchase['order']['costs'] == [0.50, 1.25]
    assert assoc_in({}, ['a', 'b', 'c'], 1) == {'a': {'b': {'c': 1}}}
    assert assoc_in({'a': 1}, ['b'], 2) == {'a': 1, 'b': 2}


def test_update_in():
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
    result = update_in(transaction, ['purchase', 'costs'], sum)
    assert result == {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': 1.75},
    }
    assert transaction['purchase']['costs'] == [0.50, 1.25]
    assert update_in({}, [1, 2, 3], str, default="bar") == {1: {2: {3: 'bar'}}}
    assert update_in({1: 'foo'}, [2, 3, 4], inc, 0) == {
        1: 'foo', 2: {3: {4: 1}}}


def test_get_in():
    transaction = {
        'name': 'Alice',
        'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
    }
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
        get_in(['a', 5], {'a': [1]}, no_default=True)
