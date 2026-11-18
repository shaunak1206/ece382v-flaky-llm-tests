import pytest
from easy_dict import NestedDict, asciify


def test_init_empty():
    nd = NestedDict()
    assert isinstance(nd, NestedDict)
    assert len(nd) == 0


def test_init_with_dict():
    d = {'a': 1, 'b': {'c': 2}}
    nd = NestedDict(d)
    assert nd['a'] == 1
    assert isinstance(nd['b'], dict)
    # NestedDict returns the dict underneath on first level


def test_getitem_existing_key():
    nd = NestedDict({'foo': 123})
    assert nd['foo'] == 123


def test_getitem_nonexisting_key_raises():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        _ = nd['missing']


def test_getitem_nested_list_key():
    data = {'a': {'b': {'c': 5}}}
    nd = NestedDict(data)
    assert nd[['a', 'b', 'c']] == 5


def test_getitem_nested_slash_key():
    data = {'x': {'y': {'z': 'val'}}}
    nd = NestedDict(data)
    assert nd['x/y/z'] == 'val'


def test_getitem_nested_key_returns_tuple():
    # If multiple matches in nested dicts: returns tuple
    data = {'root': 1, 'a': {'key': 2}, 'b': {'key': 3}}
    nd = NestedDict(data)
    val = nd['key']  # key found in both a and b nested dicts
    assert isinstance(val, tuple)
    assert 2 in val and 3 in val


def test_setitem_existing_key():
    nd = NestedDict({'foo': 1})
    nd['foo'] = 42
    assert nd['foo'] == 42


def test_setitem_new_key_at_root():
    nd = NestedDict({'foo': 1})
    nd['bar'] = 10
    assert nd['bar'] == 10


def test_setitem_nested_list_key_existing_path():
    data = {'a': {'b': {'c': 5}}}
    nd = NestedDict(data)
    nd[['a', 'b', 'c']] = 100
    assert nd['a']['b']['c'] == 100


def test_setitem_nested_slash_key_existing_path():
    data = {'a': {'b': {'c': 5}}}
    nd = NestedDict(data)
    nd['a/b/c'] = 200
    assert nd['a']['b']['c'] == 200


def test_setitem_nested_list_key_ambiguous_raises():
    data = {'a': {'c': (1, 2)}}
    nd = NestedDict(data)
    # Branches is a tuple - should raise KeyError on ambiguous keys
    with pytest.raises(KeyError):
        nd[['a', 'b']] = 10


def test_delitem_key():
    nd = NestedDict({'foo': 1, 'bar': 2})
    del nd['foo']
    assert 'foo' not in nd
    assert 'bar' in nd


def test_iter_and_len():
    d = {'x': 10, 'y': 20}
    nd = NestedDict(d)
    keys = list(iter(nd))
    assert set(keys) == set(d.keys())
    assert len(nd) == len(d)


def test_repr_contains_class_and_data():
    d = {'key': 'val'}
    nd = NestedDict(d)
    rep = repr(nd)
    assert 'NestedDict' in rep or 'easy_dict' in rep
    assert 'key' in rep


def test_call_returns_internal_dict():
    d = {'a': 1}
    nd = NestedDict(d)
    assert nd() == d


def test_contains_key():
    nd = NestedDict({'a': 1, 'b': 2})
    assert 'a' in nd
    assert 'c' not in nd


def test_anchor_creates_branch_value():
    data = {'root': {'branch1': 1}}
    nd = NestedDict({'a': {'b': {}}})
    nd.anchor('a', 'c', 100)
    assert nd['a']['c'] == 100


def test_anchor_raises_keyerror_when_trunk_not_found():
    nd = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        nd.anchor('missing', 'branch', 1)


def test_anchor_value_not_dict_and_logical_false_replaces_with_dict():
    nd = NestedDict({'a': None})
    nd.anchor('a', 'x', 123)
    assert isinstance(nd['a'], dict)
    assert nd['a']['x'] == 123


def test_setdefault_simple_key():
    nd = NestedDict({'a': 1})
    nd.setdefault('a', 10)
    nd.setdefault('b', 20)
    assert nd['a'] == 1  # existing not overwritten
    assert nd['b'] == 20  # new key set to default


def test_setdefault_nested_key_list():
    nd = NestedDict({'a': {'b': {'c': 1}}})
    nd.setdefault(['a', 'b', 'd'], 100)
    # Key d should be set to 100 under b
    assert nd['a']['b']['d'] == 100


def test_setdefault_nested_key_with_tuple():
    nd = NestedDict({'a': {'b': {}}})
    nd.setdefault(('a', 'b', 'c'), 200)
    assert nd['a']['b']['c'] == 200


def test_setdefault_raises_keyerror_empty_branch():
    nd = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        nd.setdefault(['a'], 123)  # branches empty


def test_asciify_empty_dict_returns_empty_list():
    output = asciify({})
    assert output == []


def test_asciify_basic_tree():
    d = {'a': {'b': 1, 'c': {'d': 2}}, 'e': {}}
    result = asciify(d)
    assert any('a' in line for line in result)
    assert any('b' in line for line in result)
    assert any('c' in line for line in result)
    assert any('d' in line for line in result)
    assert any('e' in line for line in result)
