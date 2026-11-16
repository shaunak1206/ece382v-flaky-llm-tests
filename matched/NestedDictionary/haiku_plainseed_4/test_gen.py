import pytest

from easy_dict import NestedDict, asciify


def test_init_without_value_is_empty():
    nd = NestedDict()
    assert len(nd) == 0
    assert nd() == {}


def test_init_copies_top_level_mapping():
    source = {'a': 1}
    nd = NestedDict(source)
    nd['a'] = 2
    assert source == {'a': 1}
    assert nd() == {'a': 2}


def test_getitem_top_level_key():
    nd = NestedDict({'a': 1, 'b': 'two'})
    assert nd['a'] == 1
    assert nd['b'] == 'two'


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd['missing']


def test_getitem_finds_key_in_nested_dict():
    nd = NestedDict({'a': {'b': 2}})
    assert nd['b'] == 2


def test_getitem_multiple_matches_returns_tuple():
    nd = NestedDict({'a': {'x': 1}, 'b': {'x': 2}})
    assert nd['x'] == (1, 2)


def test_getitem_missing_nested_key_raises_keyerror():
    nd = NestedDict({'a': {'b': 1}})
    with pytest.raises(KeyError):
        nd['zzz']


def test_getitem_slash_path():
    nd = NestedDict({'a': {'b': {'c': 3}}})
    assert nd['a/b/c'] == 3


def test_getitem_list_path():
    nd = NestedDict({'a': {'b': 5}})
    assert nd[['a', 'b']] == 5


def test_setitem_existing_top_level_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    assert nd() == {'a': 10}


def test_setitem_new_top_level_key():
    nd = NestedDict({'a': 1})
    nd['b'] = 2
    assert nd() == {'a': 1, 'b': 2}


def test_setitem_existing_nested_key():
    nd = NestedDict({'a': {'b': 1}})
    nd['b'] = 2
    assert nd() == {'a': {'b': 2}}


def test_setitem_slash_path_updates_nested_value():
    nd = NestedDict({'a': {'b': 1}})
    nd['a/b'] = 2
    assert nd() == {'a': {'b': 2}}


def test_setitem_slash_path_missing_raises_keyerror():
    nd = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        nd['a/b'] = 1


def test_delitem_removes_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert nd['b'] == 2


def test_len_iter_contains():
    nd = NestedDict({'a': 1, 'b': 2})
    assert len(nd) == 2
    assert list(iter(nd)) == ['a', 'b']
    assert 'a' in nd
    assert 'z' not in nd


def test_get_returns_default_for_missing_key():
    nd = NestedDict({'a': 1})
    assert nd.get('a') == 1
    assert nd.get('missing') is None
    assert nd.get('missing', 'fallback') == 'fallback'


def test_setdefault_adds_only_missing_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 5)
    nd.setdefault('b', 2)
    assert nd() == {'a': 1, 'b': 2}


def test_anchor_creates_branch_under_empty_dict():
    nd = NestedDict({'a': {}})
    nd.anchor('a', 'b', 5)
    assert nd() == {'a': {'b': 5}}


def test_anchor_missing_trunk_raises_keyerror():
    nd = NestedDict({'x': 1})
    with pytest.raises(KeyError):
        nd.anchor('missing', 'b', 1)


def test_anchor_non_falsy_value_raises_valueerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(ValueError):
        nd.anchor('a', 'b', 2)


def test_anchor_replaces_falsy_value():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'b', 7)
    assert nd() == {'a': {'b': 7}}


def test_asciify_empty_dict_returns_empty_list():
    assert asciify({}) == []


def test_asciify_simple_tree():
    tree = {'a': {'b': None, 'c': None}}
    assert asciify(tree) == ['a', '+-- b', '`-- c']
