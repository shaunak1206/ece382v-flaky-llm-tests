from collections.abc import MutableMapping

import pytest

from easy_dict import NestedDict, asciify


def test_init_with_values_exposes_items_and_length():
    nd = NestedDict({'a': 1, 'b': 2})
    assert len(nd) == 2
    assert nd() == {'a': 1, 'b': 2}


def test_init_copies_top_level_of_input_mapping():
    source = {'a': 1}
    nd = NestedDict(source)
    nd['z'] = 99
    assert 'z' not in source
    assert source == {'a': 1}


def test_getitem_top_level_key():
    nd = NestedDict({'a': 1})
    assert nd['a'] == 1


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_getitem_finds_key_in_nested_dict():
    nd = NestedDict({'a': {'b': 1}})
    assert nd['b'] == 1


def test_getitem_returns_tuple_when_key_matches_in_several_branches():
    nd = NestedDict({'a': {'x': {'b': 1}}, 'c': {'b': 2}})
    assert nd['b'] == (1, 2)


def test_getitem_slash_separated_path():
    nd = NestedDict({'a': {'b': {'c': 5}}})
    assert nd['a/b/c'] == 5


def test_getitem_list_path():
    nd = NestedDict({'a': {'b': 7}})
    assert nd[['a', 'b']] == 7


def test_setitem_overwrites_existing_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 2
    assert nd() == {'a': 2}


def test_setitem_adds_new_top_level_key():
    nd = NestedDict({'a': 1})
    nd['x'] = 5
    assert nd() == {'a': 1, 'x': 5}


def test_setitem_updates_existing_nested_key():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}


def test_setitem_unknown_key_goes_to_top_level_when_not_nested():
    nd = NestedDict({'a': {'b': 1}})
    nd['c'] = 3
    assert nd() == {'a': {'b': 1}, 'c': 3}


def test_setitem_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 9
    assert nd() == {'a': {'b': 9}}


def test_setitem_slash_path_with_missing_tip_raises_keyerror():
    nd = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        nd['a/zz'] = 1


def test_delitem_removes_top_level_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert nd() == {'b': 2}


def test_contains_and_iter_only_consider_top_level_keys():
    nd = NestedDict({'a': {'b': 1}, 'c': 2})
    assert 'a' in nd
    assert 'b' not in nd
    assert list(nd) == ['a', 'c']


def test_repr_shows_module_name_and_contents():
    nd = NestedDict({'a': 1})
    assert repr(nd) == "easy_dict{'a': 1}"


def test_anchor_adds_branch_to_existing_dict_value():
    nd = NestedDict({'a': {}})
    nd.anchor('a', 'b', 5)
    assert nd() == {'a': {'b': 5}}


def test_anchor_replaces_falsy_value_with_dict():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}


def test_anchor_raises_valueerror_for_truthy_non_dict_value():
    nd = NestedDict({'a': 1})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b')


def test_anchor_raises_keyerror_when_trunk_missing():
    nd = NestedDict({'x': 1})
    with pytest.raises(KeyError):
        nd.anchor('a', 'b')


def test_anchor_finds_trunk_in_nested_dict():
    nd = NestedDict({'x': {'a': None}})
    nd.anchor('a', 'b', 2)
    assert nd() == {'x': {'a': {'b': 2}}}


def test_setdefault_only_sets_missing_top_level_keys():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 5)
    nd.setdefault('b', 6)
    assert nd() == {'a': 1, 'b': 6}


def test_asciify_nested_chain_renders_tree():
    d = {'a': {'b': {'c': None}}}
    assert asciify(d) == ['a', '`-- b', '    `-- c']


def test_asciify_siblings_render_vertical_guides():
    d = {'a': {'b': {'x': None}, 'c': None}}
    assert asciify(d) == ['a', '+-- b', '|   `-- x', '`-- c']
