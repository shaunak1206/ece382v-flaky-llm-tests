import pytest

from boltons.setutils import IndexedSet, complement


def test_indexed_set_construction_and_lookup():
    x = IndexedSet(list(range(4)) + list(range(8)))
    assert list(x) == list(range(8))
    assert repr(x) == 'IndexedSet([0, 1, 2, 3, 4, 5, 6, 7])'
    assert len(x) == 8
    assert x[0] == 0
    assert x[-1] == 7
    assert 3 in x
    assert 9 not in x
    assert x.index(5) == 5
    assert x.count(3) == 1
    assert x.count(99) == 0
    assert list(reversed(x)) == list(range(7, -1, -1))
    assert len(IndexedSet()) == 0

    fcr = IndexedSet('freecreditreport.com')
    assert ''.join(fcr[:fcr.index('.')]) == 'frecditpo'

    with pytest.raises(IndexError):
        x[8]
    with pytest.raises(IndexError):
        x[-9]
    with pytest.raises(ValueError):
        x.index(99)


def test_indexed_set_slicing():
    x = IndexedSet('abcdef')
    assert isinstance(x[1:3], IndexedSet)
    assert list(x[1:3]) == ['b', 'c']
    assert list(x[:2]) == ['a', 'b']
    assert list(x[-2:]) == ['e', 'f']
    assert list(x[::2]) == ['a', 'c', 'e']
    assert list(x[:]) == list('abcdef')


def test_indexed_set_remove_discard_and_pop():
    x = IndexedSet('abcde')
    x.remove('c')
    assert list(x) == list('abde')
    assert len(x) == 4
    assert x[2] == 'd'
    assert x.index('d') == 2
    assert 'c' not in x

    x.discard('zzz')
    with pytest.raises(KeyError):
        x.remove('zzz')

    y = IndexedSet('abc')
    assert y.pop() == 'c'
    assert y.pop(0) == 'a'
    assert list(y) == ['b']

    y.add('b')
    y.add('q')
    assert list(y) == ['b', 'q']
    y.clear()
    assert len(y) == 0
    assert list(y) == []


def test_indexed_set_binary_operations():
    a = IndexedSet([1, 2, 3, 4])
    b = IndexedSet([3, 4, 5])
    assert list(a | b) == [1, 2, 3, 4, 5]
    assert list(a & b) == [3, 4]
    assert list(a - b) == [1, 2]
    assert list(a ^ b) == [1, 2, 5]
    assert list(a.union([9])) == [1, 2, 3, 4, 9]
    assert list(a & {2, 3}) == [2, 3]
    assert list(a.intersection([4, 3, 2], [3, 4])) == [3, 4]
    assert list(a.difference([1], [4])) == [2, 3]
    assert {0, 1, 9} - a == {0, 9}
    assert type({0, 1, 9} - a) is set

    assert a.isdisjoint({7, 8})
    assert not a.isdisjoint([4, 8])
    assert IndexedSet([1, 2]).issubset({1, 2, 3})
    assert not IndexedSet([1, 5]).issubset({1, 2, 3})
    assert a.issuperset([1, 2])
    assert not a.issuperset([1, 9])
    # operands are not modified
    assert list(a) == [1, 2, 3, 4]
    assert list(b) == [3, 4, 5]


def test_indexed_set_in_place_operations():
    a = IndexedSet(range(6))
    a.update([10, 11], [12])
    assert list(a) == [0, 1, 2, 3, 4, 5, 10, 11, 12]
    a.difference_update({1, 3})
    assert list(a) == [0, 2, 4, 5, 10, 11, 12]
    a.intersection_update({0, 2, 4, 99})
    assert list(a) == [0, 2, 4]

    a |= {7}
    assert set(a) == {0, 2, 4, 7}
    a -= {0}
    assert set(a) == {2, 4, 7}
    a &= {2, 7, 8}
    assert set(a) == {2, 7}
    a ^= {7, 8}
    assert set(a) == {2, 8}
    assert len(a) == 2


def test_indexed_set_sort_reverse_and_equality():
    x = IndexedSet([3, 1, 2])
    x.sort()
    assert list(x) == [1, 2, 3]
    assert x.index(3) == 2
    x.sort(reverse=True)
    assert list(x) == [3, 2, 1]
    x.reverse()
    assert list(x) == [1, 2, 3]
    assert x[0] == 1

    assert IndexedSet([1, 2]) == IndexedSet([1, 2])
    assert IndexedSet([1, 2]) != IndexedSet([2, 1])
    assert IndexedSet([1, 2]) == {2, 1}
    assert IndexedSet([1]) != 5
    assert IndexedSet.from_iterable('aab') == IndexedSet('ab')


def test_complement_membership_and_operators():
    comp = complement({2, 3})
    assert 2 not in comp
    assert 5 in comp
    assert complement(complement({1, 2})) == {1, 2}
    assert sorted(set(range(5)) & comp) == [0, 1, 4]

    universal = complement(set())
    assert 'anything' in universal
    assert bool(universal)

    c1 = complement({1})
    c2 = complement({2})
    both = c1 & c2
    assert 1 not in both and 2 not in both and 3 in both
    either = c1 | c2
    assert 1 in either and 2 in either
    minus = c1 - {5}
    assert 1 not in minus and 5 not in minus and 6 in minus

    inverted = ~c1
    assert 1 in inverted
    assert 2 not in inverted
    assert not bool(~universal)

    assert complement({1, 2}) <= complement({1})
    assert not complement({1}) <= complement({2})
    assert complement({1}) >= complement({1, 2})

    with pytest.raises(NotImplementedError):
        len(comp)
    with pytest.raises(NotImplementedError):
        iter(comp)
    with pytest.raises(NotImplementedError):
        comp.pop()
