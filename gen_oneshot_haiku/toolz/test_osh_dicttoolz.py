import pytest
from toolz.dicttoolz import (
    merge, merge_with, valmap, keymap, itemmap,
    valfilter, keyfilter, itemfilter, assoc, dissoc,
    assoc_in, update_in, get_in
)


class TestMerge:
    def test_merge_basic(self):
        result = merge({1: 'one'}, {2: 'two'})
        assert result == {1: 'one', 2: 'two'}

    def test_merge_overlapping(self):
        result = merge({1: 2, 3: 4}, {3: 3, 4: 4})
        assert result == {1: 2, 3: 3, 4: 4}

    def test_merge_empty(self):
        result = merge({})
        assert result == {}

    def test_merge_multiple(self):
        result = merge({1: 'a'}, {2: 'b'}, {3: 'c'})
        assert result == {1: 'a', 2: 'b', 3: 'c'}

    def test_merge_iterable_of_dicts(self):
        dicts = [{1: 'a'}, {2: 'b'}, {3: 'c'}]
        result = merge(dicts)
        assert result == {1: 'a', 2: 'b', 3: 'c'}


class TestMergeWith:
    def test_merge_with_sum(self):
        result = merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
        assert result == {1: 11, 2: 22}

    def test_merge_with_list_append(self):
        result = merge_with(lambda x: list(x), {1: 1, 2: 2}, {1: 10, 2: 20})
        assert result[1] == [1, 10]
        assert result[2] == [2, 20]

    def test_merge_with_max(self):
        result = merge_with(max, {1: 1, 2: 2}, {1: 10, 2: 20})
        assert result == {1: 10, 2: 20}

    def test_merge_with_multiple_dicts(self):
        result = merge_with(sum, {1: 1}, {1: 2}, {1: 3})
        assert result == {1: 6}


class TestValmap:
    def test_valmap_sum(self):
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        result = valmap(sum, bills)
        assert result == {'Alice': 65, 'Bob': 45}

    def test_valmap_double(self):
        d = {'a': 1, 'b': 2}
        result = valmap(lambda x: x * 2, d)
        assert result == {'a': 2, 'b': 4}

    def test_valmap_empty(self):
        result = valmap(lambda x: x, {})
        assert result == {}


class TestKeymap:
    def test_keymap_lower(self):
        bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
        result = keymap(str.lower, bills)
        assert result == {'alice': [20, 15, 30], 'bob': [10, 35]}

    def test_keymap_upper(self):
        d = {'a': 1, 'b': 2}
        result = keymap(str.upper, d)
        assert result == {'A': 1, 'B': 2}

    def test_keymap_empty(self):
        result = keymap(str.upper, {})
        assert result == {}


class TestItemmap:
    def test_itemmap_reversed(self):
        accountids = {"Alice": 10, "Bob": 20}
        result = itemmap(reversed, accountids)
        assert result == {10: "Alice", 20: "Bob"}

    def test_itemmap_swap_values(self):
        d = {1: 'a', 2: 'b'}
        result = itemmap(lambda kv: (kv[1], kv[0]), d)
        assert result == {'a': 1, 'b': 2}


class TestValfilter:
    def test_valfilter_even(self):
        iseven = lambda x: x % 2 == 0
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        result = valfilter(iseven, d)
        assert result == {1: 2, 3: 4}

    def test_valfilter_none(self):
        result = valfilter(lambda x: x, {'a': 0, 'b': 1, 'c': None})
        assert result == {'b': 1}


class TestKeyfilter:
    def test_keyfilter_even(self):
        iseven = lambda x: x % 2 == 0
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        result = keyfilter(iseven, d)
        assert result == {2: 3, 4: 5}

    def test_keyfilter_startswith(self):
        d = {'apple': 1, 'avocado': 2, 'banana': 3}
        result = keyfilter(lambda x: x.startswith('a'), d)
        assert result == {'apple': 1, 'avocado': 2}


class TestItemfilter:
    def test_itemfilter_basic(self):
        def isvalid(item):
            k, v = item
            return k % 2 == 0 and v < 4
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        result = itemfilter(isvalid, d)
        assert result == {2: 3}

    def test_itemfilter_custom(self):
        def is_long_key_low_val(item):
            k, v = item
            return len(k) > 2 and v < 5
        d = {'apple': 3, 'a': 1, 'banana': 10, 'cat': 2}
        result = itemfilter(is_long_key_low_val, d)
        assert result == {'apple': 3, 'cat': 2}


class TestAssoc:
    def test_assoc_new_value(self):
        result = assoc({'x': 1}, 'x', 2)
        assert result == {'x': 2}

    def test_assoc_new_key(self):
        result = assoc({'x': 1}, 'y', 3)
        assert result == {'x': 1, 'y': 3}

    def test_assoc_no_mutation(self):
        original = {'x': 1}
        result = assoc(original, 'x', 2)
        assert original == {'x': 1}
        assert result == {'x': 2}


class TestDissoc:
    def test_dissoc_single_key(self):
        result = dissoc({'x': 1, 'y': 2}, 'y')
        assert result == {'x': 1}

    def test_dissoc_multiple_keys(self):
        result = dissoc({'x': 1, 'y': 2}, 'y', 'x')
        assert result == {}

    def test_dissoc_missing_key(self):
        result = dissoc({'x': 1}, 'y')
        assert result == {'x': 1}

    def test_dissoc_no_mutation(self):
        original = {'x': 1, 'y': 2}
        result = dissoc(original, 'y')
        assert original == {'x': 1, 'y': 2}
        assert result == {'x': 1}


class TestAssocIn:
    def test_assoc_in_basic(self):
        purchase = {
            'name': 'Alice',
            'order': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]},
            'credit card': '5555-1234-1234-1234'
        }
        result = assoc_in(purchase, ['order', 'costs'], [0.25, 1.00])
        assert result['order']['costs'] == [0.25, 1.00]
        assert result['name'] == 'Alice'
        assert purchase['order']['costs'] == [0.50, 1.25]  # original unchanged

    def test_assoc_in_nested(self):
        d = {'a': {'b': {'c': 1}}}
        result = assoc_in(d, ['a', 'b', 'c'], 2)
        assert result == {'a': {'b': {'c': 2}}}


class TestUpdateIn:
    def test_update_in_increment(self):
        inc = lambda x: x + 1
        result = update_in({'a': 0}, ['a'], inc)
        assert result == {'a': 1}

    def test_update_in_nested(self):
        transaction = {
            'name': 'Alice',
            'purchase': {
                'items': ['Apple', 'Orange'],
                'costs': [0.50, 1.25]
            },
            'credit card': '5555-1234-1234-1234'
        }
        result = update_in(transaction, ['purchase', 'costs'], sum)
        assert result['purchase']['costs'] == 1.75

    def test_update_in_with_default(self):
        result = update_in({}, [1, 2, 3], str, default="bar")
        assert result == {1: {2: {3: 'bar'}}}

    def test_update_in_nested_with_default(self):
        result = update_in({1: 'foo'}, [2, 3, 4], lambda x: x + 1, 0)
        assert result == {1: 'foo', 2: {3: {4: 1}}}


class TestGetIn:
    def test_get_in_basic(self):
        transaction = {
            'name': 'Alice',
            'purchase': {
                'items': ['Apple', 'Orange'],
                'costs': [0.50, 1.25]
            },
            'credit card': '5555-1234-1234-1234'
        }
        result = get_in(['purchase', 'items', 0], transaction)
        assert result == 'Apple'

    def test_get_in_nested_key(self):
        transaction = {'name': 'Alice', 'purchase': {'total': 100}}
        result = get_in(['name'], transaction)
        assert result == 'Alice'

    def test_get_in_missing_nested(self):
        transaction = {'purchase': {'items': []}}
        result = get_in(['purchase', 'total'], transaction)
        assert result is None

    def test_get_in_missing_with_default(self):
        transaction = {'purchase': {'items': []}}
        result = get_in(['purchase', 'total'], transaction, 0)
        assert result == 0

    def test_get_in_missing_no_default_raises(self):
        transaction = {}
        with pytest.raises(KeyError):
            get_in(['y'], transaction, no_default=True)


class TestDictOperationsWithFactory:
    def test_merge_with_factory(self):
        from collections import OrderedDict
        result = merge({1: 'a'}, {2: 'b'}, factory=dict)
        assert type(result) == dict

    def test_valmap_with_factory(self):
        d = {'a': 1, 'b': 2}
        result = valmap(lambda x: x * 2, d, factory=dict)
        assert result == {'a': 2, 'b': 4}

    def test_dissoc_with_factory(self):
        result = dissoc({'x': 1, 'y': 2, 'z': 3}, 'x', factory=dict)
        assert result == {'y': 2, 'z': 3}


class TestComplexDictOperations:
    def test_nested_update_in(self):
        d = {'level1': {'level2': {'value': 5}}}
        result = update_in(d, ['level1', 'level2', 'value'], lambda x: x * 2)
        assert result == {'level1': {'level2': {'value': 10}}}

    def test_chained_operations(self):
        d = {'a': 1, 'b': 2, 'c': 3}
        # Filter values > 1, then map to double
        filtered = valfilter(lambda x: x > 1, d)
        result = valmap(lambda x: x * 2, filtered)
        assert result == {'b': 4, 'c': 6}

    def test_merge_and_filter(self):
        d1 = {1: 1, 2: 2, 3: 3}
        d2 = {3: 30, 4: 40, 5: 50}
        merged = merge(d1, d2)
        filtered = valfilter(lambda x: x > 10, merged)
        assert filtered == {3: 30, 4: 40, 5: 50}
