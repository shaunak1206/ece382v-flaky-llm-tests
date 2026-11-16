import pytest

from easy_dict import NestedDict, asciify


def test_basic_mapping_behaviour():
    nd = NestedDict({'a': 1, 'b': 2})
    assert nd['a'] == 1
    assert len(nd) == 2
    assert sorted(nd) == ['a', 'b']
    assert 'a' in nd
    assert 'z' not in nd
    assert nd() == {'a': 1, 'b': 2}


def test_empty_construction():
    nd = NestedDict()
    assert len(nd) == 0
    assert nd() == {}


def test_repr_contains_module_name_and_content():
    nd = NestedDict({'a': 1})
    assert repr(nd) == "easy_dict{'a': 1}"


def test_initial_value_top_level_is_copied():
    orig = {'a': 1}
    nd = NestedDict(orig)
    nd['b'] = 2
    assert 'b' not in orig
    assert nd['b'] == 2


def test_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['z']


def test_get_returns_default_for_missing_key():
    nd = NestedDict({'a': 1})
    assert nd.get('a') == 1
    assert nd.get('zzz') is None
    assert nd.get('zzz', 42) == 42


def test_deep_lookup_by_leaf_key():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd['c'] == 1
    assert nd['b'] == {'c': 1}


def test_lookup_with_slash_path():
    nd = NestedDict({'a': {'b': 5}})
    assert nd['a/b'] == 5


def test_lookup_with_list_path_multiple_levels():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd[['a', 'b', 'c']] == 1
    assert nd['a/b/c'] == 1
    assert nd[['a', 'b']] == {'c': 1}


def test_single_element_list_lookup():
    nd = NestedDict({'a': {'b': 1}})
    assert nd[['a']] == {'b': 1}


def test_duplicate_deep_keys_return_tuple():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_setitem_new_top_level_key():
    nd = NestedDict({'a': 1})
    nd['b'] = 2
    assert nd() == {'a': 1, 'b': 2}


def test_setitem_existing_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    assert nd['a'] == 10
    assert len(nd) == 1


def test_setitem_updates_nested_key_found_deeper():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}


def test_setitem_unknown_key_added_at_root_not_nested():
    nd = NestedDict({'a': {'b': 1}})
    nd['z'] = 3
    assert nd() == {'a': {'b': 1}, 'z': 3}


def test_setitem_with_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 5
    assert nd() == {'a': {'b': 5}}


def test_setitem_with_path_to_missing_tip_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['a/zzz'] = 1


def test_setitem_path_disambiguates_duplicate_keys():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    nd['a/x'] = 5
    assert nd()['a']['x'] == 5
    assert nd()['b']['x'] == 2


def test_delitem_and_update_and_pop():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    with pytest.raises(KeyError):
        del nd['a']
    nd.update({'c': 3})
    assert nd['c'] == 3
    assert nd.pop('c') == 3
    assert 'c' not in nd


def test_contains_only_checks_top_level():
    nd = NestedDict({'a': {'b': 1}})
    assert 'a' in nd
    assert 'b' not in nd


def test_anchor_replaces_falsy_value_with_dict():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}


def test_anchor_default_value_is_none():
    nd = NestedDict({'a': {}})
    nd.anchor('a', 'b')
    assert nd() == {'a': {'b': None}}


def test_anchor_into_existing_dict_and_nested_trunk():
    nd = NestedDict({'a': {'b': 1}, 'x': {'t': None}})
    nd.anchor('a', 'c', 2)
    nd.anchor('t', 'u', 3)
    assert nd() == {'a': {'b': 1, 'c': 2}, 'x': {'t': {'u': 3}}}


def test_anchor_errors():
    nd = NestedDict({'a': 5})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 1)
    with pytest.raises(KeyError):
        nd.anchor('missing', 'b', 1)


def test_setdefault_plain_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 100)
    nd.setdefault('b', 2)
    assert nd() == {'a': 1, 'b': 2}


def test_asciify_empty_and_single():
    assert asciify({}) == []
    assert asciify({'a': None}) == ['a']


def test_asciify_siblings_under_parent():
    d = {'a': {'b': None, 'c': None}, 'd': None}
    assert asciify(d) == ['a', '+-- b', '`-- c', 'd']


def test_asciify_deep_nesting():
    d = {'a': {'b': {'c': None}}}
    assert asciify(d) == ['a', '`-- b', '    `-- c']
