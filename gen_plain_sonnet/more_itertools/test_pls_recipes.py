from itertools import combinations
from operator import add

import pytest

from more_itertools.recipes import (
    all_equal,
    batched,
    consume,
    first_true,
    flatten,
    grouper,
    iter_except,
    iter_index,
    ncycles,
    nth,
    nth_combination,
    pad_none,
    padnone,
    partition,
    powerset,
    prepend,
    quantify,
    random_combination,
    random_combination_with_replacement,
    random_permutation,
    random_product,
    repeatfunc,
    roundrobin,
    sliding_window,
    tabulate,
    tail,
    take,
    unique,
    unique_everseen,
    unique_justseen,
)


def test_take_returns_list_and_handles_short_input():
    assert take(3, range(10)) == [0, 1, 2]
    assert take(10, range(3)) == [0, 1, 2]
    assert take(0, range(5)) == []
    it = iter(range(10))
    assert take(4, it) == [0, 1, 2, 3]
    assert next(it) == 4


def test_tabulate_applies_function_to_counting_sequence():
    assert take(4, tabulate(lambda x: x**2, -3)) == [9, 4, 1, 0]
    assert take(3, tabulate(lambda x: x + 1)) == [1, 2, 3]


def test_tail_sequence_iterator_and_errors():
    assert list(tail(3, 'ABCDEFG')) == ['E', 'F', 'G']
    assert list(tail(3, iter('ABCDEFG'))) == ['E', 'F', 'G']
    assert list(tail(10, [1, 2])) == [1, 2]
    assert list(tail(0, [1, 2])) == []
    assert list(tail(0, iter([1, 2]))) == []
    with pytest.raises(ValueError):
        tail(-1, [1, 2])


def test_consume_partial_and_full():
    it = iter(range(10))
    consume(it, 3)
    assert next(it) == 3
    consume(it)
    assert list(it) == []
    short = iter(range(3))
    consume(short, 5)
    assert list(short) == []
    untouched = iter(range(3))
    consume(untouched, 0)
    assert list(untouched) == [0, 1, 2]


def test_nth_returns_item_or_default():
    assert nth(range(10), 3) == 3
    assert nth(range(10), 20) is None
    assert nth(range(10), 20, 'zebra') == 'zebra'
    assert nth(iter('abc'), 2) == 'c'


def test_all_equal_with_and_without_key():
    assert all_equal('aaaa') is True
    assert all_equal('aaab') is False
    assert all_equal([]) is True
    assert all_equal([5]) is True
    assert all_equal('AaaA', key=str.casefold) is True
    assert all_equal([1, 2, 3], key=lambda x: x < 10) is True
    assert all_equal([1, 20, 3], key=lambda x: x < 10) is False


def test_quantify_counts_true_predicates():
    assert quantify([True, False, True]) == 2
    assert quantify(range(10), pred=lambda x: x % 2 == 0) == 5
    assert quantify([]) == 0


def test_pad_none_and_padnone_alias():
    assert take(5, pad_none(range(3))) == [0, 1, 2, None, None]
    assert padnone is pad_none
    assert take(3, pad_none([])) == [None, None, None]


def test_ncycles_repeats_sequence():
    assert list(ncycles(['a', 'b'], 3)) == ['a', 'b', 'a', 'b', 'a', 'b']
    assert list(ncycles([1, 2], 0)) == []
    assert list(ncycles(iter([1, 2]), 2)) == [1, 2, 1, 2]


def test_flatten_and_prepend():
    assert list(flatten([[0, 1], [2, 3]])) == [0, 1, 2, 3]
    assert list(flatten([])) == []
    assert list(flatten(['ab', 'cd'])) == ['a', 'b', 'c', 'd']
    assert list(prepend('0', ['1', '2', '3'])) == ['0', '1', '2', '3']
    assert list(prepend(1, [])) == [1]


def test_repeatfunc_with_and_without_times():
    assert list(repeatfunc(add, 4, 3, 5)) == [8, 8, 8, 8]
    assert list(repeatfunc(add, 0, 3, 5)) == []
    calls = []

    def counter():
        calls.append(1)
        return len(calls)

    assert take(3, repeatfunc(counter)) == [1, 2, 3]


def test_grouper_modes_and_errors():
    assert list(grouper('ABCDEF', 3)) == [('A', 'B', 'C'), ('D', 'E', 'F')]
    assert list(grouper('ABCDEFG', 3, incomplete='fill', fillvalue='x')) == [
        ('A', 'B', 'C'),
        ('D', 'E', 'F'),
        ('G', 'x', 'x'),
    ]
    assert list(grouper('ABCDEFG', 3, incomplete='ignore')) == [
        ('A', 'B', 'C'),
        ('D', 'E', 'F'),
    ]
    assert list(grouper('ABCDEF', 3, incomplete='strict')) == [
        ('A', 'B', 'C'),
        ('D', 'E', 'F'),
    ]
    with pytest.raises(ValueError):
        list(grouper('ABCDEFG', 3, incomplete='strict'))
    with pytest.raises(ValueError):
        list(grouper('ABC', 2, incomplete='bogus'))


def test_roundrobin_visits_each_iterable_in_turn():
    assert list(roundrobin('ABC', 'D', 'EF')) == ['A', 'D', 'E', 'B', 'F', 'C']
    assert list(roundrobin()) == []
    assert list(roundrobin('AB')) == ['A', 'B']
    assert list(roundrobin([1, 2, 3], [])) == [1, 2, 3]


def test_partition_splits_by_predicate():
    is_odd = lambda x: x % 2 != 0
    evens, odds = partition(is_odd, range(10))
    assert list(evens) == [0, 2, 4, 6, 8]
    assert list(odds) == [1, 3, 5, 7, 9]

    false_items, true_items = partition(None, [0, 1, False, True, '', ' '])
    assert list(false_items) == [0, False, '']
    assert list(true_items) == [1, True, ' ']

    low, high = partition(lambda x: x > 2, [1, 2, 3, 4])
    assert list(high) == [3, 4]
    assert list(low) == [1, 2]


def test_powerset_yields_all_subsets():
    assert list(powerset([1, 2, 3])) == [
        (),
        (1,),
        (2,),
        (3,),
        (1, 2),
        (1, 3),
        (2, 3),
        (1, 2, 3),
    ]
    assert list(powerset([])) == [()]
    assert list(powerset([1, 1, 0])) == [
        (),
        (1,),
        (1,),
        (0,),
        (1, 1),
        (1, 0),
        (1, 0),
        (1, 1, 0),
    ]
    assert len(list(powerset(range(6)))) == 2**6


def test_unique_everseen_preserves_order_and_supports_key():
    assert list(unique_everseen('AAAABBBCCDAABBB')) == ['A', 'B', 'C', 'D']
    assert list(unique_everseen('ABBCcAD', str.casefold)) == ['A', 'B', 'C', 'D']
    assert list(unique_everseen([[1], [2], [1]], key=tuple)) == [[1], [2]]
    with pytest.raises(TypeError):
        list(unique_everseen([[1], [2]]))


def test_unique_justseen_and_unique():
    assert list(unique_justseen('AAAABBBCCDAABBB')) == list('ABCDAB')
    assert list(unique_justseen('ABBCcAD', str.lower)) == list('ABCAD')
    assert list(unique_justseen([])) == []
    assert list(unique([[1, 2], [3, 4], [1, 2]])) == [[1, 2], [3, 4]]
    assert list(unique('ABBcCAD', str.casefold)) == ['A', 'B', 'c', 'D']
    assert list(unique('ABBcCAD', str.casefold, reverse=True)) == [
        'D',
        'c',
        'B',
        'A',
    ]


def test_iter_except_stops_on_exception():
    data = [0, 1, 2]
    assert list(iter_except(data.pop, IndexError)) == [2, 1, 0]

    data = [1, 2]
    assert list(iter_except(data.pop, IndexError, first=lambda: 99)) == [99, 2, 1]

    items = [1, 2, 3, '...', 4, 5, 6]
    func = lambda: 1 + items.pop()
    assert list(iter_except(func, (IndexError, TypeError))) == [7, 6, 5]
    assert list(iter_except(func, (IndexError, TypeError))) == [4, 3, 2]
    assert list(iter_except(func, (IndexError, TypeError))) == []


def test_first_true_default_and_predicate():
    assert first_true(range(10)) == 1
    assert first_true(range(10), pred=lambda x: x > 5) == 6
    assert first_true(range(10), default='missing', pred=lambda x: x > 9) == 'missing'
    assert first_true([0, '', None]) is None
    assert first_true([], default=7) == 7


def test_nth_combination_matches_itertools_combinations():
    for r in range(0, 6):
        expected = list(combinations(range(5), r))
        for index, combo in enumerate(expected):
            assert nth_combination(range(5), r, index) == combo
    assert nth_combination(range(5), 3, 5) == (0, 3, 4)
    assert nth_combination(range(5), 3, -1) == (2, 3, 4)
    with pytest.raises(IndexError):
        nth_combination(range(5), 3, 10)
    with pytest.raises(IndexError):
        nth_combination(range(5), 3, -11)
    with pytest.raises(ValueError):
        nth_combination(range(5), -1, 0)


def test_random_functions_respect_their_contracts():
    pool = list(range(10))
    for _ in range(25):
        perm = random_permutation(pool)
        assert sorted(perm) == pool

        part = random_permutation(pool, 3)
        assert len(part) == 3
        assert len(set(part)) == 3
        assert set(part) <= set(pool)

        combo = random_combination(pool, 4)
        assert len(combo) == 4
        assert list(combo) == sorted(combo)
        assert len(set(combo)) == 4

        cwr = random_combination_with_replacement('abc', 5)
        assert len(cwr) == 5
        assert list(cwr) == sorted(cwr)
        assert set(cwr) <= set('abc')

        prod = random_product('abc', range(4), 'XYZ')
        assert prod[0] in 'abc'
        assert prod[1] in range(4)
        assert prod[2] in 'XYZ'

        rep = random_product('ab', 'cd', repeat=2)
        assert len(rep) == 4
        assert rep[0] in 'ab' and rep[2] in 'ab'
        assert rep[1] in 'cd' and rep[3] in 'cd'

    assert random_product() == ()
    assert random_permutation([]) == ()
    assert random_combination('abc', 0) == ()
    assert random_combination('abc', 3) == ('a', 'b', 'c')
    with pytest.raises(ValueError):
        random_combination('abc', 4)


def test_batched_groups_and_strict_mode():
    assert list(batched('ABCDEFG', 3)) == [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    assert list(batched([], 3)) == []
    assert list(batched(range(6), 2, strict=True)) == [(0, 1), (2, 3), (4, 5)]
    with pytest.raises(ValueError):
        list(batched('ABCDE', 2, strict=True))
    with pytest.raises(ValueError):
        list(batched('ABC', 0))


def test_sliding_window_sizes_and_errors():
    assert list(sliding_window(range(6), 4)) == [
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    ]
    assert list(sliding_window(range(3), 4)) == []
    assert list(sliding_window('abc', 1)) == [('a',), ('b',), ('c',)]
    assert list(sliding_window('abc', 2)) == [('a', 'b'), ('b', 'c')]
    big = list(sliding_window(range(30), 25))
    assert len(big) == 6
    assert big[0] == tuple(range(25))
    assert big[-1] == tuple(range(5, 30))
    with pytest.raises(ValueError):
        sliding_window([1, 2, 3], 0)


def test_iter_index_sequences_iterators_and_bounds():
    assert list(iter_index('AABCADEAF', 'A')) == [0, 1, 4, 7]
    assert list(iter_index('AABCADEAF', 'A', 1)) == [1, 4, 7]
    assert list(iter_index('AABCADEAF', 'A', 1, 7)) == [1, 4]
    assert list(iter_index('ABCDABCD', 'AB')) == [0, 4]
    assert list(iter_index([[0, 1], [2, 3], [0, 1], [2, 3]], [0, 1])) == [0, 2]
    assert list(iter_index(iter([1, 2, 1, 3, 1]), 1, 1)) == [2, 4]
    assert list(iter_index(iter('AABCADEAF'), 'A', -3)) == [7]
    assert list(iter_index('abc', 'z')) == []
