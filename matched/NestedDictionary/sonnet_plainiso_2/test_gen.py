import pytest

import easy_dict
from easy_dict import NestedDict, asciify


def test_empty_construction():
    nd = NestedDict()
    assert len(nd) == 0
    assert list(nd) == []
    assert nd() == {}


def test_construction_with_initial_value_and_basic_access():
    nd = NestedDict({'a': 1, 'b': 2})
    assert nd['a'] == 1
    assert nd['b'] == 2
    assert len(nd) == 2
    assert set(nd) == {'a', 'b'}
    assert 'a' in nd
    assert 'zzz' not in nd


def test_initial_value_top_level_is_copied():
    original = {'a': 1}
    nd = NestedDict(original)
    nd['a'] = 2
    assert nd['a'] == 2
    assert original['a'] == 1


def test_call_returns_underlying_dict():
    nd = NestedDict({'a': 1})
    assert nd() == {'a': 1}


def test_missing_key_in_flat_dict_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_get_with_default_on_missing_key():
    nd = NestedDict({'a': 1})
    assert nd.get('missing') is None
    assert nd.get('missing', 42) == 42
    assert nd.get('a') == 1


def test_lookup_deeply_nested_key_without_path():
    nd = NestedDict({'a': {'b': {'c': 1}}, 'x': 2})
    assert nd['c'] == 1


def test_lookup_ambiguous_key_returns_tuple_of_values():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_lookup_with_slash_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd['a/b/c'] == 1
    assert nd['a/b'] == {'c': 1}


def test_lookup_with_list_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd[['a', 'b', 'c']] == 1
    assert nd[['a', 'b']] == {'c': 1}


def test_lookup_path_with_missing_trunk_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['zzz/b']


def test_setitem_existing_and_new_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    nd['b'] = 20
    assert nd['a'] == 10
    assert nd['b'] == 20
    assert len(nd) == 2


def test_setitem_updates_existing_nested_key():
    data = {'a': {'b': 1}}
    nd = NestedDict(data)
    nd['b'] = 5
    assert nd() == {'a': {'b': 5}}
    assert nd['b'] == 5


def test_setitem_new_key_is_added_at_root_when_not_found_deeper():
    nd = NestedDict({'a': {'b': 1}})
    nd['z'] = 9
    assert nd() == {'a': {'b': 1}, 'z': 9}


def test_setitem_with_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 5
    assert nd['a']['b'] == 5


def test_setitem_with_path_and_missing_tip_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['a/zzz'] = 1


def test_delitem_removes_top_level_key_and_missing_raises():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert len(nd) == 1
    with pytest.raises(KeyError):
        del nd['a']


def test_mutable_mapping_behaviour():
    nd = NestedDict({'a': 1})
    nd.update({'b': 2})
    assert nd['b'] == 2
    assert nd.pop('a') == 1
    assert dict(nd.items()) == {'b': 2}
    assert nd == {'b': 2}


def test_repr_contains_module_name_and_content():
    nd = NestedDict({'a': 1})
    assert repr(nd) == easy_dict.__name__ + str({'a': 1})


def test_anchor_adds_branch_to_top_level_trunk():
    nd = NestedDict({'a': {'x': 1}})
    nd.anchor('a', 'y', 2)
    assert nd() == {'a': {'x': 1, 'y': 2}}


def test_anchor_replaces_falsy_trunk_value_with_dict():
    nd = NestedDict({'p': {'a': None}})
    nd.anchor('a', 'b', 1)
    assert nd() == {'p': {'a': {'b': 1}}}


def test_anchor_errors():
    nd = NestedDict({'a': 5})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 1)
    with pytest.raises(KeyError):
        nd.anchor('zzz', 'b', 1)


def test_setdefault_with_simple_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 100)
    nd.setdefault('b', 200)
    assert nd['a'] == 1
    assert nd['b'] == 200


def test_asciify_empty_and_flat():
    assert asciify({}) == []
    assert asciify({'a': None, 'b': None}) == ['a', 'b']


def test_asciify_one_level_nesting():
    tree = {'a': {'b': {}, 'c': {}}, 'd': {}}
    assert asciify(tree) == ['a', '+-- b', '`-- c', 'd']


def test_asciify_deeper_nesting():
    tree = {'a': {'b': {'c': {}}, 'd': {}}}
    assert asciify(tree) == ['a', '+-- b', '|   `-- c', '`-- d']
