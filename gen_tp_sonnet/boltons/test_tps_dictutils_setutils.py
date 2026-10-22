import json

import pytest

from boltons.dictutils import (
    FrozenDict,
    ManyToMany,
    OneToOne,
    OrderedMultiDict,
    subdict,
)
from boltons.setutils import IndexedSet, complement


def test_omd_add_get_and_getlist():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.get('a') == 3
    assert omd['a'] == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('zz') == []
    assert omd.get('zz') is None
    assert len(omd) == 2
    copy = omd.getlist('a')
    copy.append(99)
    assert omd.getlist('a') == [1, 3]


def test_omd_multi_iteration_and_addlist():
    omd = OrderedMultiDict()
    omd['a'] = 1
    omd['b'] = 2
    omd.add('a', 3)
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values(multi=True) == [1, 2, 3]

    other = OrderedMultiDict([('a', -1)])
    other.addlist('a', range(3))
    assert other.getlist('a') == [-1, 0, 1, 2]


def test_omd_pop_popall_and_poplast():
    omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3), ('c', 4), ('c', 5)])
    assert omd.pop('a') == 2
    assert 'a' not in omd
    assert omd.popall('c') == [4, 5]
    assert 'c' not in omd
    assert omd.pop('zz', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        omd.pop('zz')
    with pytest.raises(KeyError):
        omd.popall('zz')

    other = OrderedMultiDict([('a', 1), ('a', 2)])
    assert other.poplast('a') == 2
    assert other.getlist('a') == [1]


def test_omd_inverted_counts_and_todict():
    omd = OrderedMultiDict([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]
    assert omd.inverted().inverted().items(multi=True) == [(0, 2), (1, 2)]

    multi = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
    assert multi.counts()['a'] == 2
    assert multi.counts()['b'] == 1
    assert multi.todict() == {'a': 2, 'b': 3}
    assert multi.todict(multi=True) == {'a': [1, 2], 'b': [3]}


def test_one_to_one_keeps_inverse_in_sync():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2

    oto['b'] = 5
    assert oto.inv[5] == 'b'
    assert 2 not in oto.inv
    oto.pop('b')
    assert 5 not in oto.inv


def test_one_to_one_unique_rejects_duplicate_values():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)
    ok = OneToOne.unique({'a': 1}, b=2)
    assert ok == {'a': 1, 'b': 2}


def test_many_to_many_tracks_both_directions():
    m2m = ManyToMany()
    m2m.add('a', 1)
    m2m.add('a', 2)
    m2m.add('b', 2)
    assert set(m2m.get('a')) == {1, 2}
    assert set(m2m.inv.get(2)) == {'a', 'b'}
    assert set(m2m.keys()) == {'a', 'b'}

    m2m.remove('a', 2)
    assert set(m2m.get('a')) == {1}
    assert set(m2m.inv.get(2)) == {'b'}

    m2m.update([('c', 9), ('c', 8)])
    assert set(m2m.get('c')) == {8, 9}
    assert m2m.get('nope') == frozenset()

    m2m.replace('c', 'd')
    assert set(m2m.get('d')) == {8, 9}
    assert m2m.get('c') == frozenset()


def test_subdict_keep_and_drop():
    assert subdict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}
    assert subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c']) == {'a': 1}
    assert subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']) == {'a': 1, 'c': 3}
    assert subdict({'a': 1}, keep=['a', 'z']) == {'a': 1}
    original = {'a': 1, 'b': 2}
    result = subdict(original)
    result['c'] = 3
    assert original == {'a': 1, 'b': 2}


def test_frozendict_is_hashable_immutable_and_updatable():
    fd = FrozenDict({'a': 1})
    assert hash(fd) == hash(FrozenDict(a=1))
    assert {fd: 'x'}[FrozenDict(a=1)] == 'x'
    with pytest.raises(TypeError):
        fd['b'] = 2
    with pytest.raises(TypeError):
        del fd['a']
    with pytest.raises(TypeError):
        fd.update(b=2)

    new = fd.updated(b=2)
    assert isinstance(new, FrozenDict)
    assert new == {'a': 1, 'b': 2}
    assert fd == {'a': 1}
    assert json.dumps(fd) == '{"a": 1}'


def test_indexedset_order_uniqueness_and_indexing():
    x = IndexedSet([3, 1, 3, 2, 1])
    assert list(x) == [3, 1, 2]
    assert len(x) == 3
    assert x[0] == 3
    assert x[-1] == 2
    assert x.index(1) == 1
    assert x.count(1) == 1
    assert x.count(9) == 0
    x.add(1)
    x.add(7)
    assert list(x) == [3, 1, 2, 7]
    assert list(x[1:3]) == [1, 2]

    y = IndexedSet(list(range(4)) + list(range(8)))
    assert list(y) == list(range(8))
    assert list(y - set(range(2))) == [2, 3, 4, 5, 6, 7]
    assert y[-1] == 7

    fcr = IndexedSet('freecreditreport.com')
    assert ''.join(fcr[:fcr.index('.')]) == 'frecditpo'


def test_indexedset_pop_remove_and_discard():
    x = IndexedSet('abc')
    assert x.pop() == 'c'
    assert x.pop(0) == 'a'
    assert list(x) == ['b']
    x.discard('zzz')
    with pytest.raises((KeyError, ValueError)):
        x.remove('zzz')
    with pytest.raises((KeyError, ValueError)):
        x.index('zzz')
    x.remove('b')
    assert len(x) == 0
    x.update('xyz')
    x.clear()
    assert list(x) == []


def test_indexedset_set_operations_sort_and_reverse():
    a = IndexedSet([1, 2, 3, 4])
    assert list(a & {3, 4, 5}) == [3, 4]
    assert list(a | {5}) == [1, 2, 3, 4, 5]
    assert list(a - {3, 4, 5}) == [1, 2]
    assert set(a ^ {4, 5}) == {1, 2, 3, 5}
    assert a.isdisjoint({9, 10})
    assert IndexedSet([1, 2]).issubset({1, 2, 3})
    assert a.issuperset({1, 2})
    assert a == IndexedSet([1, 2, 3, 4])

    b = IndexedSet([3, 1, 2])
    b.sort()
    assert list(b) == [1, 2, 3]
    b.reverse()
    assert list(b) == [3, 2, 1]
    b.update([5, 1])
    assert list(b) == [3, 2, 1, 5]


def test_complement_set():
    assert list(set(range(5)) & complement({2, 3})) == [0, 1, 4]
    comp = complement({2, 3})
    assert 2 not in comp
    assert 3 not in comp
    assert 100 in comp
