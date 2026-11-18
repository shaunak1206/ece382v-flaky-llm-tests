import pytest
from easy_dict import NestedDict


def test_init_without_initial_value():
    nd = NestedDict()
    assert isinstance(nd, NestedDict)
    assert len(nd) == 0


def test_init_with_initial_dict():
    d = {'a': 1, 'b': {'c': 2}}
    nd = NestedDict(d)
    assert nd['a'] == 1
    assert isinstance(nd['b'], dict)
    assert nd['b']['c'] == 2


def test_getitem_key_at_root():
    nd = NestedDict({'x': 10, 'y': 20})
    assert nd['x'] == 10
    assert nd['y'] == 20


def test_getitem_nested_key_list():
    data = {'a': {'b': {'c': 5}}}
    nd = NestedDict(data)
    # Access nested key as list (path)
    assert nd[['a', 'b', 'c']] == 5


def test_getitem_nested_key_slash_string():
    data = {'a': {'b': {'c': 5}}}
    nd = NestedDict(data)
    # Access nested key as slash separated string
    assert nd['a/b/c'] == 5


def test_getitem_key_not_found_raises_keyerror():
    nd = NestedDict({'foo': 1})
    with pytest.raises(KeyError):
        _ = nd['nonexistent']


def test_setitem_simple_key():
    nd = NestedDict({'a': 1})
    nd['a'] = 10
    assert nd['a'] == 10


def test_setitem_nested_key_list():
    nd = NestedDict({'a': {'b': {'c': 5}}})
    nd[['a', 'b', 'c']] = 20
    assert nd['a']['b']['c'] == 20
    assert nd[['a', 'b', 'c']] == 20


def test_setitem_nested_key_slash_string():
    nd = NestedDict({'a': {'b': {'c': 5}}})
    nd['a/b/c'] = 30
    assert nd['a']['b']['c'] == 30
    assert nd['a/b/c'] == 30


def test_setitem_nested_key_raises_keyerror_on_ambiguous_branches():
    nd = NestedDict({'a': {'b': {'c': 5}}, 'b': 7})
    with pytest.raises(KeyError):
        nd[['a', 'b']] = 10  # ambiguous keys


def test_setitem_add_new_key_to_root_dict():
    nd = NestedDict({'a': 1})
    nd['new_key'] = 42
    assert nd['new_key'] == 42


def test_delitem_removes_key():
    nd = NestedDict({'a': 1, 'b': 2})
    del nd['a']
    assert 'a' not in nd
    assert 'b' in nd


def test_len_and_iter():
    d = {'foo': 1, 'bar': 2}
    nd = NestedDict(d)
    assert len(nd) == 2
    keys = list(iter(nd))
    assert set(keys) == {'foo', 'bar'}


def test_contains_operator():
    nd = NestedDict({'key': 123})
    assert 'key' in nd
    assert 'missing' not in nd


def test_repr_contains_class_name_and_dict():
    d = {'a': 1}
    nd = NestedDict(d)
    r = repr(nd)
    assert 'easy_dict' in r
    assert str(d) in r


def test_call_returns_internal_dict():
    d = {'a': 42}
    nd = NestedDict(d)
    assert nd() == d


def test_anchor_adds_branch_to_existing_trunk():
    d = {'root': {'child1': 1}}
    nd = NestedDict(d)
    nd.anchor('root', 'child2', 2)
    assert nd['root']['child2'] == 2


def test_anchor_raises_keyerror_when_trunk_not_found():
    nd = NestedDict({'some': {'key': 1}})
    with pytest.raises(KeyError):
        nd.anchor('missing', 'branch')


def test_anchor_replaces_logical_false_with_dict():
    nd = NestedDict({'root': None})
    nd.anchor('root', 'child', 10)
    assert isinstance(nd['root'], dict)
    assert nd['root']['child'] == 10

    nd2 = NestedDict({'zero': 0})
    with pytest.raises(ValueError):
        nd2.anchor('zero', 'branch')


def test_setdefault_simple_key():
    nd = NestedDict()
    ret = nd.setdefault('a', 100)
    assert ret == 100
    assert nd['a'] == 100


def test_setdefault_nested_key_list():
    nd = NestedDict()
    nd.setdefault(['a', 'b', 'c'], 42)
    assert isinstance(nd['a'], dict)
    assert isinstance(nd['a']['b'], dict)
    # key c set to 42 on the nested NestedDict wrapper
    # It internally uses nested dict for branches
    # Because .setdefault only sets on Branches, access down is dicts
    assert nd['a']['b']['c'] == 42


def test_setdefault_raises_keyerror_with_empty_branches_list():
    nd = NestedDict()
    with pytest.raises(KeyError):
        nd.setdefault(['onlytrunk'])


def test_private___isstring_containing_char():
    assert NestedDict._NestedDict__isstring_containing_char('abc/def', '/') is True
    assert NestedDict._NestedDict__isstring_containing_char('abcdef', '/') is False
    assert NestedDict._NestedDict__isstring_containing_char(123, '/') is False


def test_asciify_basic_tree():
    d = {
        'root': {
            'branch1': {
                'leaf1': {},
                'leaf2': {}
            },
            'branch2': {
                'leaf3': {}
            }
        }
    }

    lines = NestedDict.asciify(d)
    assert any('root' in line for line in lines)
    assert any('branch1' in line for line in lines)
    assert any('leaf1' in line for line in lines)
    assert any('branch2' in line for line in lines)
    # should return list of strings
    assert all(isinstance(line, str) for line in lines)
