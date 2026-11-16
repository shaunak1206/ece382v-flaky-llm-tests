import pytest

from easy_dict import NestedDict, asciify


def test_init_empty():
    nd = NestedDict()
    assert len(nd) == 0
    assert list(nd) == []
    assert nd() == {}


def test_init_with_initial_value_and_basic_mapping_protocol():
    nd = NestedDict({'a': 1, 'b': 2})
    assert len(nd) == 2
    assert 'a' in nd
    assert 'z' not in nd
    assert list(nd) == ['a', 'b']
    assert nd() == {'a': 1, 'b': 2}


def test_initial_value_is_copied_not_aliased():
    source = {'a': 1}
    nd = NestedDict(source)
    nd['b'] = 2
    assert source == {'a': 1}


def test_getitem_root_key():
    nd = NestedDict({'a': 1, 'b': 2})
    assert nd['a'] == 1
    assert nd['b'] == 2


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_getitem_missing_key_with_nested_dicts_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['missing']


def test_getitem_deep_key_found_without_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd['c'] == 1
    assert nd['b'] == {'c': 1}


def test_getitem_multiple_matches_returns_tuple():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_getitem_with_slash_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd['a/b/c'] == 1
    assert nd['a/b'] == {'c': 1}


def test_getitem_with_list_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd[['a', 'b', 'c']] == 1
    assert nd[['a']] == {'b': {'c': 1}}


def test_setitem_existing_root_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 5
    assert nd() == {'a': 5}


def test_setitem_new_key_added_at_root():
    nd = NestedDict({'a': {'b': 1}})
    nd['z'] = 3
    assert nd() == {'a': {'b': 1}, 'z': 3}


def test_setitem_updates_existing_nested_key():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}


def test_setitem_with_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 5
    assert nd() == {'a': {'b': 5}}


def test_setitem_with_path_to_missing_tip_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['a/zzz'] = 1


def test_delitem():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert nd() == {'b': 2}
    with pytest.raises(KeyError):
        del nd['a']


def test_repr():
    nd = NestedDict({'a': 1})
    assert repr(nd) == "easy_dict{'a': 1}"


def test_anchor_replaces_falsy_value_with_dict():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}


def test_anchor_adds_branch_to_existing_dict():
    nd = NestedDict({'a': {'x': 1}})
    nd.anchor('a', 'y', 2)
    assert nd() == {'a': {'x': 1, 'y': 2}}


def test_anchor_on_nested_trunk():
    nd = NestedDict({'a': {'b': None}})
    nd.anchor('b', 'c', 1)
    assert nd() == {'a': {'b': {'c': 1}}}


def test_anchor_errors():
    nd = NestedDict({'a': 5})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 1)
    with pytest.raises(KeyError):
        nd.anchor('missing', 'b', 1)


def test_setdefault_simple_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 99)
    nd.setdefault('b', 2)
    assert nd() == {'a': 1, 'b': 2}


def test_asciify_empty_and_flat():
    assert asciify({}) == []
    assert asciify({'a': None, 'b': None}) == ['a', 'b']


def test_asciify_single_level_children():
    tree = {'a': {'b': None, 'c': None}, 'd': None}
    assert asciify(tree) == ['a', '+-- b', '`-- c', 'd']


def test_asciify_deeper_trees():
    assert asciify({'a': {'b': {'c': None}}, 'd': None}) == [
        'a', '`-- b', '    `-- c', 'd']
    assert asciify({'a': {'b': {'c': None}, 'e': None}}) == [
        'a', '+-- b', '|   `-- c', '`-- e']
