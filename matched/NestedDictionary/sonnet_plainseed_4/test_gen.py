import pytest

from easy_dict import NestedDict, asciify


def test_init_empty():
    nd = NestedDict()
    assert len(nd) == 0
    assert list(nd) == []
    assert nd() == {}


def test_init_with_initial_value_is_copied_at_top_level():
    src = {'a': 1, 'b': 2}
    nd = NestedDict(src)
    assert len(nd) == 2
    assert nd['a'] == 1
    src['c'] = 3
    assert 'c' not in nd


def test_getitem_top_level_key():
    nd = NestedDict({'a': 1, 'b': {'c': 2}})
    assert nd['a'] == 1
    assert nd['b'] == {'c': 2}


def test_getitem_finds_deeply_nested_key():
    nd = NestedDict({'a': {'b': {'c': 5}}})
    assert nd['c'] == 5


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['z']


def test_getitem_missing_key_in_nested_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['z']


def test_getitem_slash_path_and_list_path():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    assert nd['a/b/c'] == 1
    assert nd[['a', 'b', 'c']] == 1
    assert nd['a/b'] == {'c': 1}


def test_getitem_ambiguous_key_returns_tuple():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_setitem_existing_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    assert nd['a'] == 10
    assert len(nd) == 1


def test_setitem_new_key_on_empty_dict():
    nd = NestedDict()
    nd['k'] = 1
    assert nd() == {'k': 1}


def test_setitem_new_key_added_at_root():
    nd = NestedDict({'a': {'b': 1}})
    nd['z'] = 5
    assert nd() == {'a': {'b': 1}, 'z': 5}


def test_setitem_updates_nested_existing_key():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}
    assert 'b' not in nd


def test_setitem_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 5
    assert nd() == {'a': {'b': 5}}


def test_setitem_path_with_missing_tip_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['a/zzz'] = 1


def test_delitem_and_missing_delitem():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert len(nd) == 1
    with pytest.raises(KeyError):
        del nd['nope']


def test_iter_len_contains_only_top_level():
    nd = NestedDict({'a': {'b': 1}, 'c': 2})
    assert sorted(nd) == ['a', 'c']
    assert len(nd) == 2
    assert 'a' in nd
    assert 'b' not in nd


def test_call_and_repr():
    nd = NestedDict({'a': 1})
    assert nd() == {'a': 1}
    assert repr(nd) == 'easy_dict' + str({'a': 1})


def test_mutable_mapping_helpers():
    nd = NestedDict({'a': 1})
    nd.update({'x': 2})
    assert nd['x'] == 2
    assert nd.get('missing', 'default') == 'default'
    assert nd.pop('a') == 1
    assert 'a' not in nd


def test_anchor_adds_branch_to_nested_dict():
    nd = NestedDict({'a': {'b': 1}})
    nd.anchor('a', 'c', 5)
    assert nd() == {'a': {'b': 1, 'c': 5}}


def test_anchor_replaces_falsy_value_and_nested_trunk():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 1)
    assert nd() == {'a': {'b': 1}}

    nd2 = NestedDict({'x': {'a': None}})
    nd2.anchor('a', 'b', 2)
    assert nd2() == {'x': {'a': {'b': 2}}}


def test_anchor_errors():
    nd = NestedDict({'a': 5})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 1)
    with pytest.raises(KeyError):
        nd.anchor('missing', 'b', 1)


def test_setdefault_plain_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 9)
    nd.setdefault('b', 2)
    assert nd['a'] == 1
    assert nd['b'] == 2


def test_asciify_empty_and_flat():
    assert asciify({}) == []
    assert asciify({'a': None, 'b': None}) == ['a', 'b']


def test_asciify_simple_tree():
    assert asciify({'a': {'b': None, 'c': None}}) == ['a', '+-- b', '`-- c']


def test_asciify_deeper_tree():
    tree = {'a': {'b': {'c': None}, 'd': None}}
    assert asciify(tree) == ['a', '+-- b', '|   `-- c', '`-- d']
