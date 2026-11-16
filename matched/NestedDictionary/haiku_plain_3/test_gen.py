import pytest

from easy_dict import NestedDict, asciify


def test_init_copies_top_level_of_initial_mapping():
    source = {'a': 1}
    nd = NestedDict(source)
    source['b'] = 2
    assert nd() == {'a': 1}


def test_getitem_top_level_key():
    nd = NestedDict({'a': 1})
    assert nd['a'] == 1


def test_getitem_missing_flat_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_getitem_finds_nested_key():
    nd = NestedDict({'a': {'b': 2}})
    assert nd['b'] == 2


def test_getitem_list_path():
    nd = NestedDict({'a': {'b': {'c': 3}}})
    assert nd[['a', 'b', 'c']] == 3


def test_getitem_slash_path():
    nd = NestedDict({'a': {'b': {'c': 3}}})
    assert nd['a/b/c'] == 3


def test_setitem_existing_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 2
    assert nd['a'] == 2
    assert nd() == {'a': 2}


def test_setitem_new_key_goes_to_root_when_absent_everywhere():
    nd = NestedDict({'a': {'b': 1}})
    nd['z'] = 5
    assert nd() == {'a': {'b': 1}, 'z': 5}


def test_setitem_updates_existing_nested_key():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}
    assert nd['b'] == 2


def test_setitem_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 9
    assert nd() == {'a': {'b': 9}}


def test_setitem_slash_path_missing_tip_raises_keyerror():
    nd = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        nd['a/x'] = 1


def test_delitem_removes_top_level_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert nd() == {'b': 2}


def test_mapping_iter_len_and_contains():
    nd = NestedDict({'a': 1, 'b': 2})
    assert set(nd) == {'a', 'b'}
    assert len(nd) == 2
    assert 'a' in nd
    assert 'z' not in nd


def test_repr_uses_module_name_and_internal_dict():
    nd = NestedDict({'a': 1})
    assert repr(nd) == NestedDict.__module__ + str({'a': 1})


def test_pop_returns_value_and_removes_key():
    nd = NestedDict({'a': 1, 'b': 2})
    assert nd.pop('a') == 1
    assert 'a' not in nd


def test_anchor_creates_branch_from_none_value():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 5)
    assert nd() == {'a': {'b': 5}}


def test_anchor_creates_branch_from_empty_dict():
    nd = NestedDict({'a': {}})
    nd.anchor('a', 'b', 5)
    assert nd() == {'a': {'b': 5}}


def test_anchor_on_nested_trunk():
    nd = NestedDict({'x': {'a': None}})
    nd.anchor('a', 'b', 7)
    assert nd() == {'x': {'a': {'b': 7}}}


def test_anchor_truthy_non_dict_trunk_raises_valueerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b')


def test_anchor_missing_trunk_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd.anchor('z', 'b')


def test_setdefault_sets_only_missing_flat_key():
    nd = NestedDict()
    nd.setdefault('a', 1)
    nd.setdefault('a', 2)
    assert nd() == {'a': 1}


def test_asciify_chain_of_two_levels():
    assert asciify({'a': {'b': None}}) == ['a', '`-- b']


def test_asciify_empty_dict_returns_empty_list():
    assert asciify({}) == []
