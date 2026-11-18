import pytest
from easy_dict import NestedDict


def test_init_empty():
    d = NestedDict()
    assert len(d) == 0


def test_init_with_dict():
    d = NestedDict({'a': 1, 'b': 2})
    assert d['a'] == 1
    assert d['b'] == 2


def test_getitem_simple_key():
    d = NestedDict({'x': 100})
    assert d['x'] == 100
    with pytest.raises(KeyError):
        _ = d['y']


def test_getitem_nested_key_as_list():
    d = NestedDict({'a': {'b': {'c': 5}}})
    assert d[['a', 'b', 'c']] == 5
    assert d[['a', 'b']] == {'c': 5}
    assert d[['a']] == {'b': {'c': 5}}


def test_getitem_nested_key_as_slash_string():
    d = NestedDict({'a': {'b': {'c': 42}}})
    assert d['a/b/c'] == 42
    assert d['a/b'] == {'c': 42}
    assert d['a'] == {'b': {'c': 42}}


def test_setitem_simple_key():
    d = NestedDict({'a': 1})
    d['a'] = 42
    assert d['a'] == 42


def test_setitem_nested_key_as_list():
    d = NestedDict({'a': {'b': 1}})
    d[['a', 'b']] = 99
    assert d['a']['b'] == 99
    # Setting non-existent nested key raises KeyError
    with pytest.raises(KeyError):
        d[['a', 'c']] = 100


def test_setitem_nested_key_as_slash_string():
    d = NestedDict({'x': {'y': 10}})
    d['x/y'] = 20
    assert d['x']['y'] == 20
    with pytest.raises(KeyError):
        d['x/z'] = 30


def test_delitem_removes_key():
    d = NestedDict({'a': 1, 'b': 2})
    del d['a']
    assert 'a' not in d
    with pytest.raises(KeyError):
        del d['a']


def test_contains_key():
    d = NestedDict({'x': 1})
    assert 'x' in d
    assert 'y' not in d


def test_len_and_iter():
    d = NestedDict({'one': 1, 'two': 2, 'three': 3})
    assert len(d) == 3
    keys = list(iter(d))
    assert set(keys) == {'one', 'two', 'three'}


def test_repr_contains_class_name_and_data():
    d = NestedDict({'a': 1})
    rep = repr(d)
    assert 'NestedDict' in rep
    assert "'a': 1" in rep or '"a": 1' in rep


def test_call_returns_underlying_dict():
    data = {'k': 'v'}
    d = NestedDict(data)
    assert d() == data


def test_anchor_normal_anchors_value():
    d = NestedDict({'root': {'foo': 1}})
    d.anchor('root', 'bar', 2)
    assert d['root']['bar'] == 2


def test_anchor_nonexistent_root_key_raises_keyerror():
    d = NestedDict({'x': {}})
    with pytest.raises(KeyError):
        d.anchor('nonexistent', 'branch', 123)


def test_anchor_non_dict_value_raises_valueerror():
    d = NestedDict({'a': 10})
    with pytest.raises(ValueError):
        d.anchor('a', 'branch', 5)


def test_setdefault_simple_key():
    d = NestedDict({'a': 1})
    d.setdefault('a', 100)
    d.setdefault('b', 200)
    assert d['a'] == 1
    assert d['b'] == 200


def test_setdefault_nested_key_list_existing_branch():
    d = NestedDict({'a': {'b': {'c': 1}}})
    d.setdefault(['a', 'b', 'd'], 5)
    assert d['a']['b']['d'] == 5


def test_setdefault_nested_key_list_new_branch():
    d = NestedDict({'a': None})
    d.setdefault(['a', 'b'], 10)
    assert d['a']['b'] == 10


def test_setdefault_empty_branches_raises_keyerror():
    d = NestedDict({'a': {}})
    with pytest.raises(KeyError):
        d.setdefault(['a'])


def test_getitem_returns_tuple_if_multiple_nested_keys():
    d = NestedDict({'root': {'a': 1}, 'other': {'a': 2}})
    result = d['a']
    # It returns tuple of values found in nested dicts if multiple matches
    assert isinstance(result, tuple)
    assert 1 in result
    assert 2 in result


def test_setitem_ambiguous_keys_raises_keyerror():
    d = NestedDict({'root': {'a': (1, 2)}})
    with pytest.raises(KeyError):
        d[['root', 'a']] = 100


def test_isstring_containing_char_static_method():
    # Test for the static method __isstring_containing_char
    func = NestedDict._NestedDict__isstring_containing_char  # name mangled access
    assert func('a/b/c', '/')
    assert not func('abc', '/')
    assert not func(123, '/')


def test_getitem_deep_raises_keyerror():
    d = NestedDict({'a': {'b': {}}})
    with pytest.raises(KeyError):
        _ = d['x']


def test_setitem_deep_new_key_creates_at_root():
    d = NestedDict({'a': 1})
    d['new'] = 5
    assert d['new'] == 5
