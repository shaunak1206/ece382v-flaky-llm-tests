"""LLM-generated unit tests for toolz.dicttoolz, based on source/docstrings only."""
import pytest

from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap, valfilter, keyfilter,
    itemfilter, assoc, dissoc, assoc_in, update_in, get_in,
)


def test_merge_basic_precedence_and_iterable_form():
    assert merge({1: 'one'}, {2: 'two'}) == {1: 'one', 2: 'two'}
    # later dicts win on key conflicts
    assert merge({1: 2, 3: 4}, {3: 3, 4: 4}) == {1: 2, 3: 3, 4: 4}
    # a single iterable of dicts is also accepted
    assert merge([{1: 1}, {2: 2}]) == {1: 1, 2: 2}


def test_merge_with_sum():
    result = merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    assert result == {1: 11, 2: 22}


def test_merge_with_disjoint_keys():
    result = merge_with(sum, {1: 1, 2: 2}, {2: 20, 3: 30})
    assert result == {1: 1, 2: 22, 3: 30}


def test_valmap_and_keymap():
    bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    assert valmap(sum, bills) == {'Alice': 65, 'Bob': 45}
    assert keymap(str.lower, {'ALICE': 1, 'BOB': 2}) == {'alice': 1, 'bob': 2}


def test_itemmap():
    accountids = {"Alice": 10, "Bob": 20}
    result = itemmap(reversed, accountids)
    assert result == {10: "Alice", 20: "Bob"}


def test_valfilter_keyfilter_itemfilter():
    iseven = lambda x: x % 2 == 0
    d = {1: 2, 2: 3, 3: 4, 4: 5}
    assert valfilter(iseven, d) == {1: 2, 3: 4}
    assert keyfilter(iseven, d) == {2: 3, 4: 5}

    def isvalid(item):
        k, v = item
        return k % 2 == 0 and v < 4
    assert itemfilter(isvalid, d) == {2: 3}


def test_assoc_new_and_existing_key():
    assert assoc({'x': 1}, 'x', 2) == {'x': 2}
    result = assoc({'x': 1}, 'y', 3)
    assert result == {'x': 1, 'y': 3}


def test_assoc_does_not_mutate_original():
    original = {'x': 1}
    assoc(original, 'y', 2)
    assert original == {'x': 1}


def test_dissoc():
    assert dissoc({'x': 1, 'y': 2}, 'y') == {'x': 1}
    assert dissoc({'x': 1, 'y': 2}, 'y', 'x') == {}
    assert dissoc({'x': 1}, 'y') == {'x': 1}  # ignores missing keys


def test_assoc_in_nested():
    purchase = {'name': 'Alice',
                'order': {'items': ['Apple', 'Orange'],
                          'costs': [0.50, 1.25]}}
    result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
    assert result == {
        'name': 'Alice',
        'order': {'items': ['Apple', 'Orange'], 'costs': [0.25, 1.00]},
    }
    # original untouched
    assert purchase['order']['costs'] == [0.50, 1.25]


def test_update_in_existing_key():
    inc = lambda x: x + 1
    assert update_in({'a': 0}, ['a'], inc) == {'a': 1}


def test_update_in_creates_nested_dicts_with_default():
    result = update_in({}, [1, 2, 3], str, default="bar")
    assert result == {1: {2: {3: 'bar'}}}


def test_update_in_partial_path_exists():
    inc = lambda x: x + 1
    result = update_in({1: 'foo'}, [2, 3, 4], inc, 0)
    assert result == {1: 'foo', 2: {3: {4: 1}}}


def test_get_in_found_and_missing():
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises():
    with pytest.raises(KeyError):
        get_in(['y'], {}, no_default=True)
