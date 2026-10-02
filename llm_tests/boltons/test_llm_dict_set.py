"""Unit tests for boltons.dictutils and boltons.setutils."""
import copy
import pickle

import pytest

from boltons.dictutils import (
    FrozenDict, FrozenHashError, ManyToMany, OMD, OneToOne, OrderedMultiDict,
    subdict)
from boltons.setutils import IndexedSet, complement


# ------------------------------------------------------------ OrderedMultiDict

@pytest.fixture
def omd():
    return OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])


def test_omd_get_returns_latest_and_getlist_returns_all(omd):
    assert omd['a'] == 3
    assert omd.get('a') == 3
    assert omd.getlist('a') == [1, 3]
    assert omd.getlist('missing') == []
    assert omd.get('missing', 'dflt') == 'dflt'
    assert len(omd) == 2


def test_omd_repr_shows_all_pairs_in_insertion_order(omd):
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"


def test_omd_getlist_returns_a_copy(omd):
    vals = omd.getlist('a')
    vals.append(99)
    assert omd.getlist('a') == [1, 3]


def test_omd_setitem_replaces_all_values_for_key(omd):
    omd['a'] = 9
    assert omd.getlist('a') == [9]
    assert omd.items(multi=True) == [('b', 2), ('a', 9)]


def test_omd_add_and_addlist_preserve_existing_values():
    omd = OrderedMultiDict()
    omd.add('k', 1)
    omd.addlist('k', iter([2, 3]))
    omd.addlist('empty', [])
    assert omd.getlist('k') == [1, 2, 3]
    assert 'empty' not in omd


def test_omd_items_keys_values_single_and_multi(omd):
    assert omd.items() == [('a', 3), ('b', 2)]
    assert omd.items(multi=True) == [('a', 1), ('b', 2), ('a', 3)]
    assert omd.keys() == ['a', 'b']
    assert omd.keys(multi=True) == ['a', 'b', 'a']
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]


def test_omd_reversed_iterates_keys_from_the_end(omd):
    assert list(reversed(omd)) == ['b', 'a']


def test_omd_pop_poplast_and_popall(omd):
    assert omd.poplast('a') == 3
    assert omd.getlist('a') == [1]
    assert omd.pop('a') == 1
    assert 'a' not in omd
    other = OrderedMultiDict([('x', 1), ('x', 2), ('y', 3)])
    assert other.popall('x') == [1, 2]
    assert other.poplast() == 3
    assert len(other) == 0


def test_omd_pop_errors_and_defaults():
    omd = OrderedMultiDict()
    with pytest.raises(KeyError):
        omd.pop('nope')
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.pop('nope', 'dflt') == 'dflt'
    assert omd.popall('nope', []) == []
    assert omd.poplast(default=None) is None


def test_omd_update_overwrites_while_update_extend_appends(omd):
    overwritten = omd.copy()
    overwritten.update({'a': 10})
    assert overwritten.items(multi=True) == [('b', 2), ('a', 10)]
    extended = omd.copy()
    extended.update_extend({'a': 10})
    assert extended.items(multi=True) == [('a', 1), ('b', 2), ('a', 3), ('a', 10)]


def test_omd_update_with_pair_iterable_keeps_duplicates_from_input(omd):
    omd.update([('a', 7), ('a', 8)])
    assert omd.getlist('a') == [7, 8]
    assert omd['b'] == 2


def test_omd_todict_flat_and_multi(omd):
    assert omd.todict() == {'a': 3, 'b': 2}
    assert omd.todict(multi=True) == {'a': [1, 3], 'b': [2]}


def test_omd_inverted_counts_and_sorted():
    omd = OMD([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]
    assert omd.inverted().inverted() == omd
    assert OMD([('a', 1), ('a', 2), ('b', 3)]).counts().todict() == {'a': 2, 'b': 1}
    assert OMD(zip(range(3), range(3))).sorted(reverse=True).items() == [(2, 2), (1, 1), (0, 0)]


def test_omd_sortedvalues_sorts_within_each_key():
    omd = OrderedMultiDict()
    omd.addlist('even', [6, 2])
    omd.addlist('odd', [1, 5])
    omd.add('even', 4)
    omd.add('odd', 3)
    sorted_omd = omd.sortedvalues()
    assert sorted_omd.getlist('even') == [2, 4, 6]
    assert sorted_omd.getlist('odd') == [1, 3, 5]
    assert sorted_omd.keys(multi=True) == omd.keys(multi=True)


def test_omd_equality_is_order_sensitive_but_matches_plain_dict(omd):
    assert omd == omd.copy()
    assert omd != OrderedMultiDict([('b', 2), ('a', 1), ('a', 3)])
    assert omd == {'a': 3, 'b': 2}
    assert omd != 'string'


def test_omd_copy_is_independent(omd):
    clone = omd.copy()
    clone.add('c', 4)
    assert 'c' not in omd
    assert clone.getlist('a') == omd.getlist('a')


def test_omd_pickle_roundtrip_keeps_multiple_values(omd):
    restored = pickle.loads(pickle.dumps(omd))
    assert restored == omd
    assert restored.getlist('a') == [1, 3]


def test_omd_constructor_rules_and_fromkeys():
    assert OrderedMultiDict(a=1)['a'] == 1
    with pytest.raises(TypeError):
        OrderedMultiDict([], [])
    assert OrderedMultiDict.fromkeys(['x', 'y'], 0).todict() == {'x': 0, 'y': 0}


def test_omd_setdefault_delete_and_clear(omd):
    assert omd.setdefault('a', 100) == 3
    assert omd.setdefault('new') is None
    assert 'new' in omd
    del omd['a']
    assert omd.keys(multi=True) == ['b', 'new']
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []


# ---------------------------------------------------------------- OneToOne

def test_one_to_one_maintains_inverse_mapping():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[1] == 'a'
    assert len(oto) == 2


def test_one_to_one_overwrite_through_inverse():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert len(oto) == 2


def test_one_to_one_setting_taken_value_evicts_previous_key():
    oto = OneToOne({'a': 1, 'b': 2})
    oto['a'] = 2
    assert dict(oto) == {'a': 2}
    assert dict(oto.inv) == {2: 'a'}


def test_one_to_one_duplicate_values_resolve_to_last_key():
    oto = OneToOne({'a': 1, 'b': 1})
    assert dict(oto) == {'b': 1}
    assert oto.inv == {1: 'b'}


def test_one_to_one_unique_constructor_rejects_duplicates():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)
    assert OneToOne.unique({'a': 1}, b=2)['b'] == 2


def test_one_to_one_pop_delete_clear_and_copy():
    oto = OneToOne({'a': 1, 'b': 2, 'c': 3})
    assert oto.pop('a') == 1
    assert 1 not in oto.inv
    assert oto.pop('zzz', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        oto.pop('zzz')
    del oto['b']
    assert 2 not in oto.inv
    clone = oto.copy()
    oto.clear()
    assert len(oto) == 0 and len(oto.inv) == 0
    assert clone == {'c': 3}
    assert clone.inv == {3: 'c'}


def test_one_to_one_update_and_unhashable_values():
    oto = OneToOne()
    oto.update([('x', 1), ('y', 2)], z=3)
    assert oto.inv[3] == 'z'
    with pytest.raises(TypeError):
        oto['bad'] = []


# ----------------------------------------------------------------- FrozenDict

def test_frozen_dict_is_hashable_and_usable_as_key():
    fd = FrozenDict({'a': 1, 'b': 2})
    assert hash(fd) == hash(FrozenDict({'b': 2, 'a': 1}))
    lookup = {fd: 'found'}
    assert lookup[FrozenDict(a=1, b=2)] == 'found'
    assert fd == {'a': 1, 'b': 2}


@pytest.mark.parametrize('mutator', [
    lambda d: d.__setitem__('x', 1),
    lambda d: d.__delitem__('a'),
    lambda d: d.update({'x': 1}),
    lambda d: d.setdefault('x', 1),
    lambda d: d.pop('a'),
    lambda d: d.popitem(),
    lambda d: d.clear(),
])
def test_frozen_dict_rejects_mutation(mutator):
    fd = FrozenDict({'a': 1})
    with pytest.raises(TypeError):
        mutator(fd)
    assert fd == {'a': 1}


def test_frozen_dict_updated_returns_new_instance():
    fd = FrozenDict({'a': 1})
    new = fd.updated({'b': 2}, c=3)
    assert isinstance(new, FrozenDict)
    assert new == {'a': 1, 'b': 2, 'c': 3}
    assert fd == {'a': 1}


def test_frozen_dict_unhashable_values_raise_on_hash():
    fd = FrozenDict({'a': [1, 2]})
    with pytest.raises(FrozenHashError):
        hash(fd)
    with pytest.raises(TypeError):
        hash(fd)


def test_frozen_dict_copy_pickle_fromkeys_and_repr():
    fd = FrozenDict.fromkeys('ab', 0)
    assert fd == {'a': 0, 'b': 0}
    assert copy.copy(fd) is fd
    assert pickle.loads(pickle.dumps(fd)) == fd
    assert repr(fd).startswith('FrozenDict(')


# ------------------------------------------------------- subdict / ManyToMany

def test_subdict_keep_and_drop():
    d = {'a': 1, 'b': 2, 'c': 3}
    assert subdict(d) == d
    assert subdict(d, keep=['a', 'c']) == {'a': 1, 'c': 3}
    assert subdict(d, drop=['b', 'c']) == {'a': 1}
    assert subdict(d, keep=['a', 'zzz']) == {'a': 1}
    assert subdict(d, keep=['a', 'b'], drop=['b']) == {'a': 1}


def test_many_to_many_tracks_both_directions():
    m2m = ManyToMany()
    m2m.add('alice', 'python')
    m2m.add('alice', 'rust')
    m2m.add('bob', 'python')
    assert m2m['alice'] == frozenset({'python', 'rust'})
    assert m2m.inv['python'] == frozenset({'alice', 'bob'})
    m2m.remove('alice', 'python')
    assert m2m.inv['python'] == frozenset({'bob'})
    assert 'alice' in m2m and len(m2m) == 2
    assert m2m.get('nobody') == frozenset()


def test_many_to_many_setitem_and_delitem_keep_inverse_in_sync():
    m2m = ManyToMany([('a', 1), ('a', 2)])
    m2m['a'] = [2, 3]
    assert m2m['a'] == frozenset({2, 3})
    assert 1 not in m2m.inv
    del m2m['a']
    assert len(m2m) == 0 and len(m2m.inv) == 0


# ------------------------------------------------------------------ IndexedSet

def test_indexed_set_dedupes_and_keeps_insertion_order():
    iset = IndexedSet(list(range(4)) + list(range(8)))
    assert list(iset) == list(range(8))
    assert iset == IndexedSet(range(8))
    assert repr(iset) == 'IndexedSet([0, 1, 2, 3, 4, 5, 6, 7])'


def test_indexed_set_indexing_negative_and_out_of_range():
    iset = IndexedSet('abcde')
    assert iset[0] == 'a'
    assert iset[-1] == 'e'
    with pytest.raises(IndexError):
        iset[10]
    with pytest.raises(IndexError):
        iset[-10]


def test_indexed_set_slicing_returns_indexed_set():
    iset = IndexedSet('abcdef')
    assert iset[1:4] == IndexedSet('bcd')
    assert isinstance(iset[::2], IndexedSet)
    assert list(iset[::2]) == ['a', 'c', 'e']
    assert list(iset[::-1]) == list('fedcba')


def test_indexed_set_doc_example_with_string_set():
    fcr = IndexedSet('freecreditreport.com')
    assert ''.join(fcr[:fcr.index('.')]) == 'frecditpo'
    assert list(fcr & set('cash4gold.com')) == ['c', 'd', 'o', '.', 'm']


def test_indexed_set_remove_and_discard_keep_indices_consistent():
    iset = IndexedSet('abcdef')
    iset.remove('c')
    iset.discard('zzz')
    iset.remove('a')
    assert list(iset) == list('bdef')
    assert len(iset) == 4
    assert iset[1] == 'd'
    assert iset.index('f') == 3
    assert 'c' not in iset
    with pytest.raises(KeyError):
        iset.remove('c')


def test_indexed_set_pop_from_end_and_by_index():
    iset = IndexedSet('abcd')
    assert iset.pop() == 'd'
    assert iset.pop(0) == 'a'
    assert list(iset) == ['b', 'c']
    assert iset.pop(1) == 'c'


def test_indexed_set_index_and_count():
    iset = IndexedSet('xyz')
    assert iset.index('z') == 2
    assert iset.count('x') == 1
    assert iset.count('q') == 0
    with pytest.raises(ValueError):
        iset.index('q')


def test_indexed_set_sort_and_reverse_in_place():
    iset = IndexedSet([3, 1, 2])
    iset.sort()
    assert list(iset) == [1, 2, 3]
    iset.reverse()
    assert list(iset) == [3, 2, 1]
    iset.sort(reverse=True)
    assert list(iset) == [3, 2, 1]
    assert iset.index(1) == 2


def test_indexed_set_operators_follow_set_semantics():
    a = IndexedSet([1, 2, 3, 4])
    assert list(a | [5, 1]) == [1, 2, 3, 4, 5]
    assert list(a & {2, 4, 9}) == [2, 4]
    assert list(a - {1, 2}) == [3, 4]
    assert list(a ^ {4, 5}) == [1, 2, 3, 5]
    assert a == {1, 2, 3, 4}


def test_indexed_set_in_place_operators():
    a = IndexedSet([1, 2, 3, 4])
    a |= {5}
    a &= {1, 2, 5}
    assert list(a) == [1, 2, 5]
    a -= {2}
    assert list(a) == [1, 5]
    a ^= {5, 6}
    assert list(a) == [1, 6]


def test_indexed_set_subset_superset_disjoint():
    a = IndexedSet([1, 2, 3])
    assert a.issubset({1, 2, 3, 4})
    assert not a.issubset({1, 2})
    assert a.issuperset({1, 2})
    assert not a.issuperset({1, 9})
    assert a.isdisjoint({7, 8})
    assert not a.isdisjoint({3})


def test_indexed_set_clear_update_and_from_iterable():
    iset = IndexedSet.from_iterable('hello')
    assert list(iset) == ['h', 'e', 'l', 'o']
    iset.update('world', [1])
    assert list(iset) == ['h', 'e', 'l', 'o', 'w', 'r', 'd', 1]
    iset.clear()
    assert len(iset) == 0 and list(iset) == []


def test_indexed_set_many_removals_still_consistent():
    iset = IndexedSet(range(1000))
    for i in range(0, 1000, 2):
        iset.remove(i)
    assert len(iset) == 500
    assert iset[0] == 1
    assert iset[-1] == 999
    assert iset.index(501) == 250
    assert list(iset)[:3] == [1, 3, 5]


# ------------------------------------------------------------------ complement

def test_complement_intersection_with_regular_set():
    assert sorted(set(range(5)) & complement({2, 3})) == [0, 1, 4]


def test_complement_membership():
    comp = complement({2, 3})
    assert 1 in comp
    assert 2 not in comp


def test_complement_of_complement_is_original():
    s = set(range(5))
    assert complement(complement(s)) == s


def test_complement_has_no_defined_length_or_iteration():
    comp = complement(set())
    with pytest.raises(NotImplementedError):
        len(comp)
    with pytest.raises(NotImplementedError):
        list(comp)
    assert bool(comp)


def test_complement_add_and_remove_adjust_exclusions():
    comp = complement({1, 2})
    comp.add(1)
    assert 1 in comp
    comp.remove(5)
    assert 5 not in comp


def test_complement_rejects_non_set_operands():
    with pytest.raises(TypeError):
        complement({1}).intersection([1, 2])
