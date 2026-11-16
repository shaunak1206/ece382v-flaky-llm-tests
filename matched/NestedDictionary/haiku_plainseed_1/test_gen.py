import pytest

from easy_dict import NestedDict, asciify


def test_empty_dict_has_no_items():
    nd = NestedDict()
    assert len(nd) == 0
    assert list(nd) == []


def test_initial_value_is_copied_not_shared():
    source = {'a': 1}
    nd = NestedDict(source)
    nd['a'] = 5
    assert source == {'a': 1}
    assert nd['a'] == 5


def test_root_key_lookup_returns_value():
    nd = NestedDict({'a': 1, 'b': 'two'})
    assert nd['a'] == 1
    assert nd['b'] == 'two'


def test_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_missing_key_in_empty_dict_raises_keyerror():
    nd = NestedDict()
    with pytest.raises(KeyError):
        nd['anything']


def test_key_inside_nested_dict_is_found_without_path():
    nd = NestedDict({'a': {'b': 2}})
    assert nd['b'] == 2


def test_key_deeply_nested_is_found_without_path():
    nd = NestedDict({'a': {'b': {'c': 3}}})
    assert nd['c'] == 3


def test_duplicate_keys_in_branches_return_tuple_of_values():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_slash_separated_path_lookup():
    nd = NestedDict({'a': {'b': 2}})
    assert nd['a/b'] == 2


def test_list_path_lookup():
    nd = NestedDict({'a': {'b': {'c': 3}}})
    assert nd[['a', 'b', 'c']] == 3


def test_setitem_overwrites_existing_root_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    assert nd() == {'a': 10}


def test_setitem_adds_new_root_key():
    nd = NestedDict({'a': 1})
    nd['new'] = 2
    assert nd() == {'a': 1, 'new': 2}


def test_setitem_updates_existing_nested_key_in_place():
    nd = NestedDict({'a': {'b': 2}})
    nd['b'] = 9
    assert nd() == {'a': {'b': 9}}


def test_setitem_adds_new_key_at_root_when_not_found_nested():
    nd = NestedDict({'a': {'b': 2}})
    nd['c'] = 3
    assert nd() == {'a': {'b': 2}, 'c': 3}


def test_setitem_with_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 2}})
    nd['a/b'] = 7
    assert nd() == {'a': {'b': 7}}


def test_delitem_removes_root_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert nd() == {'b': 2}


def test_len_iter_and_contains_reflect_root_keys():
    nd = NestedDict({'a': 1, 'b': 2})
    assert len(nd) == 2
    assert list(nd) == ['a', 'b']
    assert 'a' in nd
    assert 'z' not in nd


def test_repr_shows_module_name_and_contents():
    nd = NestedDict({'a': 1})
    assert repr(nd) == "easy_dict{'a': 1}"


def test_anchor_creates_branch_under_existing_empty_dict():
    nd = NestedDict({'a': {}})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}


def test_anchor_replaces_falsy_none_trunk_with_dict():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}


def test_anchor_raises_valueerror_for_truthy_non_dict_trunk():
    nd = NestedDict({'a': 1})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 2)


def test_anchor_raises_keyerror_for_missing_trunk():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd.anchor('zz', 'b', 2)


def test_setdefault_adds_missing_key():
    nd = NestedDict()
    nd.setdefault('a', 1)
    assert nd() == {'a': 1}


def test_setdefault_keeps_existing_value():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 99)
    assert nd() == {'a': 1}


def test_asciify_renders_sibling_branches():
    result = asciify({'a': {'x': None, 'y': None}})
    assert result == ['a', '+-- x', '`-- y']


def test_asciify_renders_nested_chain():
    result = asciify({'a': {'b': {'c': None}}, 'd': None})
    assert result == ['a', '`-- b', '    `-- c', 'd']
