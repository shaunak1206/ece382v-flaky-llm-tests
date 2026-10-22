from collections import OrderedDict

from toolz import (
    assoc, assoc_in, dissoc, get_in, itemfilter, itemmap, keyfilter, keymap,
    merge, merge_with, update_in, valfilter, valmap,
)


def inc(x):
    return x + 1


def test_merge_basic():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}


def test_merge_later_wins():
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}


def test_merge_accepts_list_of_dicts_and_does_not_mutate():
    a = {1: 1}
    b = {2: 2}
    assert merge([a, b]) == {1: 1, 2: 2}
    assert a == {1: 1}
    assert b == {2: 2}


def test_merge_with_sum():
    assert merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20}) == {1: 11, 2: 22}


def test_merge_with_collects_lists():
    assert merge_with(list, {1: 1, 2: 2}, {2: 20, 3: 30}) == {
        1: [1], 2: [2, 20], 3: [30]}


def test_valmap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {'Alice': 65, 'Bob': 45}


def test_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert keymap(str.lower, bills) == {'alice': [20, 15, 30], 'bob': [10, 35]}


def test_itemmap():
    ids = {"Alice": 10, "Bob": 20}
    assert itemmap(lambda kv: (kv[1], kv[0]), ids) == {10: "Alice", 20: "Bob"}


def test_filters():
    def iseven(x):
        return x % 2 == 0

    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4

    assert itemfilter(isvalid, d) == {2: 3}


def test_assoc_does_not_mutate():
    d = {'x': 1}
    assert assoc(d, 'x', 2) == {'x': 2}
    assert assoc(d, 'y', 3) == {'x': 1, 'y': 3}
    assert d == {'x': 1}


def test_dissoc():
    d = {'x': 1, 'y': 2}
    assert dissoc(d, 'y') == {'x': 1}
    assert dissoc(d, 'y', 'x') == {}
    assert dissoc({'x': 1}, 'y') == {'x': 1}
    assert d == {'x': 1, 'y': 2}


def test_assoc_in():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
                'credit card': '5555-1234-1234-1234'}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
        'credit card': '5555-1234-1234-1234',
    }
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_update_in():
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_missing_key_uses_default():
    assert update_in({}, ['a', 'b'], inc, default=0) == {'a': {'b': 1}}


def test_get_in():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_default_and_no_default():
    import pytest

    d = {'a': {'b': 1}}
    assert get_in(['a', 'x'], d, default=0) == 0
    assert get_in(['a', 'b'], d) == 1
    with pytest.raises(KeyError):
        get_in(['a', 'x'], d, no_default=True)
