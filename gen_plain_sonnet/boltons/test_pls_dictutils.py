import copy
import pickle

import pytest

from boltons.dictutils import (
    FrozenDict,
    ManyToMany,
    OneToOne,
    OrderedMultiDict,
    subdict,
)


def _sample():
    omd = OrderedMultiDict()
    omd.add('a', 1)
    omd.add('b', 2)
    omd.add('a', 3)
    return omd


def test_omd_add_and_lookup():
    omd = _sample()
    assert len(omd) == 2
    assert 'a' in omd
    assert omd['a'] == 3
    assert omd.get('a') == 3
    assert omd.get('z') is None
    assert omd.get('z', 'dflt') == 'dflt'
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('z') == []
    assert omd.getlist('z', default=None) is None

    copy_of_list = omd.getlist('a')
    copy_of_list.append(99)
    assert omd.getlist('a') == [1, 3]

    omd.addlist('c', range(3))
    assert omd.getlist('c') == [0, 1, 2]
    omd.addlist('empty', [])
    assert 'empty' not in omd
    with pytest.raises(KeyError):
        omd['missing']


def test_omd_ordering_views():
    omd = _sample()
    assert omd.keys() == ['a', 'b']
    assert omd.values() == [3, 2]
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert list(omd) == ['a', 'b']
    assert omd.todict() == {'a': 3, 'b': 2}
    assert omd.todict(multi=True) == {'a': [1, 3], 'b': [2]}
    assert list(reversed(omd)) == ['b', 'a']
    assert repr(OrderedMultiDict([(1, 2)])) == 'OrderedMultiDict([(1, 2)])'


def test_omd_setitem_update_and_setdefault():
    omd = _sample()
    omd['a'] = 9
    assert omd.getlist('a') == [9]
    # re-setting a key moves it to the end of the insertion order
    assert omd.keys() == ['b', 'a']

    omd = _sample()
    omd.update({'a': 7})
    assert omd.getlist('a') == [7]

    omd = OrderedMultiDict([('b', 1)])
    omd.update([('b', 5), ('b', 6)], c=1)
    assert omd.getlist('b') == [5, 6]
    assert omd['c'] == 1

    omd = _sample()
    omd.update_extend({'a': 7})
    assert omd.getlist('a') == [1, 3, 7]

    assert omd.setdefault('new', 'v') == 'v'
    assert omd.setdefault('new', 'other') == 'v'
    assert omd.setdefault('none_default') is None
    assert 'none_default' in omd

    omd = _sample()
    omd |= {'b': 20}
    assert omd.getlist('b') == [20]


def test_omd_pop_family():
    omd = _sample()
    assert omd.pop('a') == 3
    assert 'a' not in omd
    assert omd.items(multi=True) == [('b', 2)]
    with pytest.raises(KeyError):
        omd.pop('a')
    assert omd.pop('a', 'dflt') == 'dflt'

    omd = _sample()
    assert omd.popall('a') == [1, 3]
    assert omd.popall('a', 'gone') == 'gone'
    with pytest.raises(KeyError):
        omd.popall('a')

    omd = _sample()
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.poplast('a') == 1
    assert 'a' not in omd
    assert omd.poplast() == 2
    assert len(omd) == 0
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default='empty') == 'empty'

    omd = _sample()
    del omd['a']
    assert omd.items(multi=True) == [('b', 2)]
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []


def test_omd_inverted_counts_sorted_copy_and_eq():
    assert OrderedMultiDict([(0, 2), (1, 2)]).inverted().getlist(2) == [0, 1]
    omd = _sample()
    assert omd.inverted().inverted() == omd
    assert omd.counts().todict() == {'a': 2, 'b': 1}

    ordered = OrderedMultiDict(zip(range(3), range(3)))
    assert ordered.sorted(reverse=True).items(multi=True) == [(2, 2), (1, 1), (0, 0)]

    somd = omd.sortedvalues(reverse=True)
    assert somd.getlist('a') == [3, 1]
    assert somd.keys(multi=True) == omd.keys(multi=True)

    dup = omd.copy()
    assert dup == omd
    assert dup is not omd
    dup.add('z', 1)
    assert dup != omd
    assert 'z' not in omd

    other_order = OrderedMultiDict([('b', 2), ('a', 1), ('a', 3)])
    assert other_order != omd
    assert omd == {'a': 3, 'b': 2}
    assert omd != 5
    assert OrderedMultiDict.fromkeys(['x', 'y'], 0).items() == [('x', 0), ('y', 0)]
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])


def test_omd_pickle_and_copy_preserve_multivalues():
    omd = _sample()
    restored = pickle.loads(pickle.dumps(omd))
    assert restored == omd
    assert restored.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert restored.getlist('a') == [1, 3]

    shallow = copy.copy(omd)
    assert shallow.items(multi=True) == omd.items(multi=True)


def test_one_to_one():
    mapping = OneToOne({'a': 1, 'b': 2})
    assert mapping.inv == {1: 'a', 2: 'b'}
    assert mapping.inv.inv is mapping

    mapping['c'] = 3
    assert mapping.inv[3] == 'c'
    del mapping['a']
    assert 1 not in mapping.inv

    # assigning an already-used value evicts the previous owner
    mapping['b'] = 3
    assert mapping == {'b': 3}
    assert mapping.inv == {3: 'b'}

    assert mapping.pop('b') == 3
    assert mapping.inv == {}
    assert mapping.pop('b', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        mapping.pop('b')

    mapping.update({'x': 10, 'y': 20}, z=30)
    assert mapping.inv == {10: 'x', 20: 'y', 30: 'z'}
    assert mapping.setdefault('x', 99) == 10
    assert mapping.copy() == mapping
    mapping.clear()
    assert len(mapping) == 0 and len(mapping.inv) == 0

    # duplicate values: the last key wins unless unique() is used
    assert OneToOne({'a': 1, 'b': 1}) == {'b': 1}
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)
    with pytest.raises(TypeError):
        mapping['k'] = []


def test_many_to_many():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.add('b', 2)
    assert m2m['a'] == frozenset({1, 2})
    assert m2m.inv[2] == frozenset({'a', 'b'})
    assert len(m2m) == 2
    assert 'a' in m2m
    assert sorted(m2m.iteritems()) == [('a', 1), ('a', 2), ('b', 2)]

    m2m.remove('a', 2)
    assert m2m['a'] == frozenset({1})
    assert m2m.inv[2] == frozenset({'b'})

    del m2m['b']
    assert 'b' not in m2m
    assert 2 not in m2m.inv

    m2m.replace('a', 'z')
    assert 'a' not in m2m
    assert m2m['z'] == frozenset({1})
    assert m2m.inv[1] == frozenset({'z'})
    assert m2m.get('missing') == frozenset()
    with pytest.raises(KeyError):
        m2m['missing']

    m2m['k'] = [1, 2]
    m2m['k'] = [2, 3]
    assert m2m['k'] == frozenset({2, 3})
    assert 1 in m2m.inv  # still owned by 'z'
    assert m2m.inv[3] == frozenset({'k'})

    built = ManyToMany([('x', 1), ('x', 2), ('y', 1)])
    assert built.inv[1] == frozenset({'x', 'y'})
    assert sorted(built.keys()) == ['x', 'y']
    assert built == ManyToMany([('x', 1), ('x', 2), ('y', 1)])
    assert built != ManyToMany([('x', 1)])


def test_subdict_and_frozendict():
    base = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(base) == base
    assert subdict(base, drop=['b', 'c']) == {'a': 1}
    assert subdict(base, keep=['a', 'c']) == {'a': 1, 'c': 3}
    assert subdict(base, keep=['a', 'z']) == {'a': 1}
    assert subdict(base, keep=['a', 'b'], drop=['b']) == {'a': 1}
    assert base == {'a': 1, 'b': 2, 'c': 3}

    frozen = FrozenDict({'a': 1})
    assert hash(frozen) == hash(FrozenDict(a=1))
    assert {frozen: 'x'}[FrozenDict(a=1)] == 'x'
    assert repr(frozen) == "FrozenDict({'a': 1})"
    assert copy.copy(frozen) is frozen
    assert pickle.loads(pickle.dumps(frozen)) == frozen

    with pytest.raises(TypeError):
        frozen['b'] = 2
    with pytest.raises(TypeError):
        del frozen['a']
    with pytest.raises(TypeError):
        frozen.update(b=2)
    with pytest.raises(TypeError):
        frozen.pop('a')
    with pytest.raises(TypeError):
        frozen.clear()
    with pytest.raises(TypeError):
        frozen.setdefault('b', 1)

    bigger = frozen.updated(b=2)
    assert bigger == {'a': 1, 'b': 2}
    assert type(bigger) is FrozenDict
    assert frozen == {'a': 1}
    assert FrozenDict.fromkeys(['a', 'b'], 0) == {'a': 0, 'b': 0}

    with pytest.raises(TypeError):
        hash(FrozenDict({'a': [1]}))
