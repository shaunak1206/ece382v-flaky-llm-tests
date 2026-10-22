from itertools import count, permutations, takewhile

import pytest

from more_itertools.more import (
    bucket,
    chunked,
    collapse,
    consumer,
    derangements,
    distinct_permutations,
    first,
    ilen,
    interleave,
    interleave_evenly,
    interleave_longest,
    interleave_randomly,
    intersperse,
    iterate,
    last,
    nth_or_last,
    one,
    peekable,
    sized_iterator,
    spy,
    strictly_n,
    substrings,
    substrings_indexes,
    unique_to_each,
    windowed,
    with_iter,
)
from more_itertools.recipes import multinomial, roundrobin, take


def test_chunked_basic():
    assert list(chunked([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]
    assert list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3)) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]
    assert list(chunked([], 3)) == []
    assert list(chunked('abcde', 2)) == [['a', 'b'], ['c', 'd'], ['e']]
    assert list(chunked(range(3), None)) == [[0, 1, 2]]


def test_chunked_strict_and_invalid_sizes():
    assert list(chunked([1, 2, 3, 4], 2, strict=True)) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3, 4, 5], 2, strict=True))
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3], -1))
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3], None, strict=True))


def test_first_last_and_nth_or_last():
    assert first([0, 1, 2, 3]) == 0
    assert first([], 'some default') == 'some default'
    with pytest.raises(ValueError):
        first([])
    it = iter(range(5))
    assert first(it) == 0
    assert next(it) == 1

    assert last([0, 1, 2, 3]) == 3
    assert last('abc') == 'c'
    assert last(iter(range(5))) == 4
    assert last([], 'some default') == 'some default'
    with pytest.raises(ValueError):
        last([])
    with pytest.raises(ValueError):
        last(range(0))

    assert nth_or_last([0, 1, 2, 3], 2) == 2
    assert nth_or_last([0, 1], 2) == 1
    assert nth_or_last([], 0, 'some default') == 'some default'
    with pytest.raises(ValueError):
        nth_or_last([], 3)


def test_peekable_peek_prepend_and_bool():
    p = peekable(['a', 'b'])
    assert p
    assert p.peek() == 'a'
    assert next(p) == 'a'
    assert p.peek() == 'b'
    assert next(p) == 'b'
    assert not p
    assert p.peek('hi') == 'hi'
    with pytest.raises(StopIteration):
        p.peek()
    with pytest.raises(StopIteration):
        next(p)

    p = peekable([1, 2, 3])
    p.prepend(10, 11, 12)
    assert next(p) == 10
    assert p.peek() == 11
    assert list(p) == [11, 12, 1, 2, 3]

    p = peekable([])
    p.prepend(1)
    assert next(p) == 1
    with pytest.raises(StopIteration):
        next(p)


def test_peekable_indexing_and_slicing():
    p = peekable('abcde')
    assert p[0] == 'a'
    assert p[2] == 'c'
    assert next(p) == 'a'
    assert p[-1] == 'e'
    assert list(p) == ['b', 'c', 'd', 'e']

    p = peekable(range(10))
    assert p[2:5] == [2, 3, 4]
    assert next(p) == 0
    assert p[::-1] == list(range(9, 0, -1))
    assert next(p) == 1

    with pytest.raises(IndexError):
        peekable('ab')[5]
    with pytest.raises(ValueError):
        peekable('abc')[::0]


def test_iterate_and_ilen():
    assert take(5, iterate(lambda x: x * 2, 1)) == [1, 2, 4, 8, 16]

    def step(x):
        if x >= 3:
            raise StopIteration
        return x + 1

    assert list(iterate(step, 0)) == [0, 1, 2, 3]

    assert ilen(range(10)) == 10
    assert ilen(x for x in 'abc') == 3
    assert ilen([]) == 0
    it = iter(range(4))
    assert ilen(it) == 4
    assert list(it) == []


def test_consumer_with_iter_and_sized_iterator():
    @consumer
    def collector(store):
        while True:
            store.append((yield))

    store = []
    c = collector(store)
    c.send('a')
    c.send('b')
    assert store == ['a', 'b']

    class Manager:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return [1, 2, 3]

        def __exit__(self, *exc):
            self.closed = True
            return False

    manager = Manager()
    assert list(with_iter(manager)) == [1, 2, 3]
    assert manager.closed

    manager = Manager()
    gen = with_iter(manager)
    assert next(gen) == 1
    assert not manager.closed
    gen.close()
    assert manager.closed

    sized = sized_iterator(map(str, range(5)), 5)
    assert len(sized) == 5
    assert list(sized) == ['0', '1', '2', '3', '4']


def test_one_returns_single_item_or_raises():
    assert one([7]) == 7
    assert one(x for x in [5]) == 5
    with pytest.raises(ValueError, match='too few'):
        one([])
    with pytest.raises(ValueError, match='Expected exactly one'):
        one([1, 2])
    with pytest.raises(IndexError):
        one([], too_short=IndexError('too few items'))
    with pytest.raises(RuntimeError):
        one(['too', 'many'], too_long=RuntimeError)


def test_strictly_n_validates_length():
    assert list(strictly_n(['a', 'b', 'c', 'd'], 4)) == ['a', 'b', 'c', 'd']
    with pytest.raises(ValueError, match='Too few'):
        list(strictly_n('ab', 3))
    with pytest.raises(ValueError, match='Too many'):
        list(strictly_n('abc', 2))

    seen_short = []
    seen_long = []
    assert list(strictly_n('abcdef', 4, too_long=seen_long.append)) == list('abcd')
    assert seen_long == [5]
    assert list(strictly_n('ab', 4, too_short=seen_short.append)) == ['a', 'b']
    assert seen_short == [2]


def test_distinct_permutations():
    assert list(distinct_permutations([1, 0, 1])) == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert sorted(distinct_permutations([1, 0, 1], r=2)) == [(0, 1), (1, 0), (1, 1)]
    assert sorted(distinct_permutations(range(3), r=2)) == sorted(permutations(range(3), 2))
    assert list(distinct_permutations([])) == [()]
    assert list(distinct_permutations([1, 2], r=0)) == [()]
    assert list(distinct_permutations([1, 2], r=3)) == []

    items = [1, 1, 2, 3]
    result = list(distinct_permutations(items))
    assert len(result) == len(set(result)) == 12
    assert set(result) == set(permutations(items))
    assert len(list(distinct_permutations('aabbc'))) == multinomial(2, 2, 1)

    mixed = list(distinct_permutations([1, 'a']))
    assert set(mixed) == {(1, 'a'), ('a', 1)}
    assert len(mixed) == 2


def test_derangements():
    assert list(derangements(range(3))) == [(1, 2, 0), (2, 0, 1)]
    assert len(list(derangements(range(4)))) == 9
    assert len(list(derangements(range(5)))) == 44
    assert sorted(derangements(range(3), 2)) == [(1, 0), (1, 2), (2, 0)]
    assert list(derangements(['Alice', 'Bob', 'Bob'])) == [
        ('Bob', 'Bob', 'Alice'),
        ('Bob', 'Alice', 'Bob'),
    ]
    for d in derangements('abcd'):
        assert all(x != y for x, y in zip(d, 'abcd'))


def test_intersperse_and_unique_to_each():
    assert list(intersperse('!', [1, 2, 3, 4, 5])) == [1, '!', 2, '!', 3, '!', 4, '!', 5]
    assert list(intersperse(None, [1, 2, 3, 4, 5], n=2)) == [1, 2, None, 3, 4, None, 5]
    assert list(intersperse('-', [])) == []
    assert list(intersperse('-', [1])) == [1]
    with pytest.raises(ValueError):
        intersperse('-', [1, 2], n=0)

    assert unique_to_each({'A', 'B'}, {'B', 'C'}, {'B', 'D'}) == [['A'], ['C'], ['D']]
    assert unique_to_each('mississippi', 'missouri') == [['p', 'p'], ['o', 'u', 'r']]
    assert unique_to_each() == []


def test_windowed():
    assert list(windowed([1, 2, 3, 4, 5], 3)) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    assert list(windowed([1, 2, 3], 4)) == [(1, 2, 3, None)]
    assert list(windowed([1, 2, 3], 5, fillvalue=0)) == [(1, 2, 3, 0, 0)]
    assert list(windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2)) == [
        (1, 2, 3),
        (3, 4, 5),
        (5, 6, '!'),
    ]
    assert list(windowed([], 3)) == []
    assert list(windowed(range(4), 1)) == [(0,), (1,), (2,), (3,)]
    with pytest.raises(ValueError):
        list(windowed([1, 2], 0))
    with pytest.raises(ValueError):
        list(windowed([1, 2], 2, step=0))


def test_substrings_and_substrings_indexes():
    assert [''.join(s) for s in substrings('more')] == [
        'm', 'o', 'r', 'e', 'mo', 'or', 're', 'mor', 'ore', 'more',
    ]
    assert list(substrings([0, 1, 2])) == [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]
    assert list(substrings('')) == []

    assert list(substrings_indexes('abc')) == [
        ('a', 0, 1),
        ('b', 1, 2),
        ('c', 2, 3),
        ('ab', 0, 2),
        ('bc', 1, 3),
        ('abc', 0, 3),
    ]
    reversed_items = list(substrings_indexes('abc', reverse=True))
    assert reversed_items[0] == ('abc', 0, 3)
    assert reversed_items[-1] == ('c', 2, 3)
    assert sorted(reversed_items) == sorted(substrings_indexes('abc'))


def test_bucket_and_spy():
    data = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'b3']
    s = bucket(data, key=lambda x: x[0])
    assert 'a' in s
    assert 'z' not in s
    assert list(s['b']) == ['b1', 'b2', 'b3']
    assert list(s['a']) == ['a1', 'a2']
    assert sorted(s) == ['a', 'b', 'c']
    assert list(s['c']) == ['c1', 'c2']

    s = bucket(data, key=lambda x: x[0])
    first_a = s['a']
    assert next(first_a) == 'a1'
    assert next(first_a) == 'a2'
    with pytest.raises(StopIteration):
        next(first_a)

    odd_only = bucket(count(1, 2), key=lambda x: x % 10, validator=lambda x: x in {1, 3, 5, 7, 9})
    assert 2 not in odd_only
    assert list(odd_only[2]) == []
    assert next(odd_only[3]) == 3

    head, rest = spy('abcdefg')
    assert head == ['a']
    assert list(rest) == list('abcdefg')
    (x, y), rest = spy('abcdefg', 2)
    assert (x, y) == ('a', 'b')
    assert list(rest) == list('abcdefg')
    head, rest = spy([1, 2, 3], 10)
    assert head == [1, 2, 3]
    assert list(rest) == [1, 2, 3]


def test_interleave_variants():
    assert list(interleave([1, 2, 3], [4, 5], [6, 7, 8])) == [1, 4, 6, 2, 5, 7]
    assert list(interleave_longest([1, 2, 3], [4, 5], [6, 7, 8])) == [1, 4, 6, 2, 5, 7, 3, 8]
    assert list(interleave_longest([None, 1], [2])) == [None, 2, 1]
    assert list(interleave()) == []
    assert list(interleave_longest()) == []
    assert list(interleave_longest('ABC', 'D', 'EF')) == list(roundrobin('ABC', 'D', 'EF'))

    assert list(interleave_evenly([[1, 2, 3, 4, 5], ['a', 'b']])) == [1, 2, 'a', 3, 4, 'b', 5]
    assert list(interleave_evenly([[1, 2, 3], [4, 5], [6, 7, 8]])) == [1, 6, 4, 2, 7, 3, 8, 5]
    assert list(interleave_evenly([])) == []
    lengths = [3, 2]
    assert list(interleave_evenly([iter('abc'), iter('de')], lengths=lengths)) == [
        'a', 'd', 'b', 'c', 'e',
    ]
    with pytest.raises(ValueError):
        list(interleave_evenly([iter('abc'), iter('de')]))
    with pytest.raises(ValueError):
        list(interleave_evenly([[1], [2]], lengths=[1]))

    out = list(interleave_randomly([1, 2, 3], 'abc', [10, 20, 30]))
    assert len(out) == 9
    assert [x for x in out if isinstance(x, int) and x < 10] == [1, 2, 3]
    assert [x for x in out if isinstance(x, str)] == ['a', 'b', 'c']
    assert [x for x in out if isinstance(x, int) and x >= 10] == [10, 20, 30]
    assert list(interleave_randomly()) == []


def test_collapse_flattens_nested_iterables():
    assert list(collapse([(1, 2), ([3, 4], [[5], [6]])])) == [1, 2, 3, 4, 5, 6]
    assert list(collapse(['ab', ('cd', 'ef'), ['gh', 'ij']], base_type=tuple)) == [
        'ab',
        ('cd', 'ef'),
        'gh',
        'ij',
    ]
    nested = [('a', ['b']), ('c', ['d'])]
    assert list(collapse(nested)) == ['a', 'b', 'c', 'd']
    assert list(collapse(nested, levels=1)) == ['a', ['b'], 'c', ['d']]
    assert list(collapse([[1], [2]], levels=0)) == [[1], [2]]
    assert list(collapse('abc')) == ['abc']
    assert list(collapse([b'xy', [b'z']])) == [b'xy', b'z']
    assert list(collapse(5)) == [5]
    assert list(collapse([])) == []
    assert list(takewhile(lambda x: x < 3, collapse([[0], [1], [2], [3]]))) == [0, 1, 2]
