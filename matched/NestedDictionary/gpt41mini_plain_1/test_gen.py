import pytest
from easy_dict import NestedDict


def test_init_empty():
    nd = NestedDict()
    assert isinstance(nd, NestedDict)
    assert len(nd) == 0


def test_init_with_dict():
    d = {'a': 1, 'b': {'c': 2}}
    nd = NestedDict(d)
    assert len(nd) == 2
    assert nd['a'] == 1
    # accessing nested dict should return NestedDict view or equivalent dict
    assert isinstance(nd['b'], dict)
    assert nd['b']['c'] == 2


def test_getitem_existing_key():
    nd = NestedDict({'a': 10})
    assert nd['a'] == 10


def test_getitem_nonexistent_key_raises():
    nd = NestedDict({'a': 10})
    with pytest.raises(KeyError):
        _ = nd['z']


def test_getitem_nested_key_list_access():
    d = {'x': {'y': {'z': 100}}}
    nd = NestedDict(d)
    # Using list as key: ['x', 'y', 'z']
    assert nd[['x', 'y', 'z']] == 100
    # Using list with one key just accesses at root level
    assert nd[['x']] == d['x']


def test_getitem_nested_key_slash_access():
    d = {'x': {'y': {'z': 123}}}
    nd = NestedDict(d)
    # Using slash '/' notation
    assert nd['x/y/z'] == 123
    assert nd['x'] == d['x']


def test_setitem_existing_root_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 2
    assert nd['a'] == 2


def test_setitem_nested_key_list():
    d = {'x': {'y': 1}}
    nd = NestedDict(d)
    nd[['x', 'y']] = 10
    assert nd['x']['y'] == 10


def test_setitem_nested_key_slash():
    d = {'x': {'y': 2}}
    nd = NestedDict(d)
    nd['x/y'] = 20
    assert nd['x']['y'] == 20


def test_setitem_nonexistent_key_adds_at_root():
    nd = NestedDict({'a': 1})
    # When setting a key not existing deeper or root, it adds at root
    nd['b'] = 5
    assert nd['b'] == 5


def test_setitem_list_key_ambiguous_raises():
    d = {'x': {'y': 1}}
    nd = NestedDict(d)
    # Setting a nested list key but with branch tuple inside (simulate ambiguous keys)
    with pytest.raises(KeyError):
        nd[('x', ('y',))] = 10


def test_delitem_removes_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert 'b' in nd


def test_contains():
    nd = NestedDict({'a': 1})
    assert 'a' in nd
    assert 'b' not in nd


def test_len_and_iter():
    data = {'a': 1, 'b': 2}
    nd = NestedDict(data)
    assert len(nd) == 2
    keys = set(iter(nd))
    assert keys == set(data.keys())


def test_repr_contains_class_and_data():
    nd = NestedDict({'a': 1})
    rep = repr(nd)
    assert rep.startswith('easy_dict')
    assert "'a'" in rep or "a" in rep


def test_call_returns_underlying_dict():
    d = {'a': 1}
    nd = NestedDict(d)
    val = nd()
    assert isinstance(val, dict)
    assert val['a'] == 1


def test_anchor_adds_branch_to_trunk():
    d = {'root': {}}
    nd = NestedDict(d)
    nd.anchor('root', 'branch', 123)
    assert 'branch' in nd['root']
    assert nd['root']['branch'] == 123


def test_anchor_trunk_not_exist_raises_keyerror():
    nd = NestedDict({'a': 1})
    with pytest.raises(KeyError):
        nd.anchor('nonexistent', 'branch', 5)


def test_anchor_trunk_exists_but_not_dict_value_error():
    nd = NestedDict({'a': 10})
    with pytest.raises(ValueError):
        nd.anchor('a', 'branch', 5)


def test_setdefault_simple_key_new():
    nd = NestedDict()
    nd.setdefault('foo', 42)
    assert nd['foo'] == 42


def test_setdefault_simple_key_existing():
    nd = NestedDict({'foo': 10})
    nd.setdefault('foo', 42)
    assert nd['foo'] == 10  # existing value remains


def test_setdefault_nested_key_list():
    nd = NestedDict()
    nd.setdefault(['a', 'b'], 99)
    assert 'a' in nd
    assert isinstance(nd['a'], dict)
    assert nd['a']['b'] == 99


def test_setdefault_nested_key_tuple():
    nd = NestedDict()
    nd.setdefault(('a', 'b', 'c'), 100)
    assert 'a' in nd
    assert isinstance(nd['a'], dict)
    assert 'b' in nd['a']
    assert isinstance(nd['a']['b'], dict)
    assert nd['a']['b']['c'] == 100


def test_setdefault_no_branches_keyerror():
    nd = NestedDict()
    with pytest.raises(KeyError):
        nd.setdefault([])


def test_private_isstring_containing_char():
    # _NestedDict__isstring_containing_char is static method, access via class
    method = NestedDict._NestedDict__isstring_containing_char
    assert method('a/b', '/') is True
    assert method('abc', '/') is False
    assert method(['a/b'], '/') is False
    assert method('', '/') is False


def test_asciify_simple_dict():
    d = {'a': {'b': {'c': {}}}}
    result = NestedDict.asciify(d)
    # Result is list of strings representing ascii tree
    assert isinstance(result, list)
    assert any('a' in line for line in result)
    assert any('b' in line for line in result)
    assert any('c' in line for line in result)


def test_asciify_empty_dict_returns_empty_list():
    assert NestedDict.asciify({}) == []


def test_asciify_multiple_branches():
    d = {
        'root': {
            'branch1': {},
            'branch2': {
                'leaf': {}
            }
        }
    }
    lines = NestedDict.asciify(d)
    assert any('root' in line for line in lines)
    assert any('branch1' in line for line in lines)
    assert any('branch2' in line for line in lines)
    assert any('leaf' in line for line in lines)
