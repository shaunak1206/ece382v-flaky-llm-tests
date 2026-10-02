import itertools
import random
from operator import add, mul

import pytest

from toolz import (
    accumulate,
    concat,
    count,
    diff,
    drop,
    first,
    frequencies,
    get,
    groupby,
    interleave,
    isdistinct,
    iterate,
    join,
    last,
    mapcat,
    merge_sorted,
    nth,
    partition,
    partition_all,
    peek,
    peekn,
    pluck,
    random_sample,
    reduceby,
    remove,
    second,
    sliding_window,
    tail,
    take,
    topk,
    unique,
)


def iseven(x):
    return x % 2 == 0


def test_groupby_with_function():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    assert groupby(len, names) == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7], True: [2, 4, 6, 8]}


def test_groupby_with_member_key_and_empty():
    people = [
        {'name': 'Alice', 'gender': 'F'},
        {'name': 'Bob', 'gender': 'M'},
        {'name': 'Charlie', 'gender': 'M'},
    ]
    assert groupby('gender', people) == {
        'F': [{'name': 'Alice', 'gender': 'F'}],
        'M': [{'name': 'Bob', 'gender': 'M'},
              {'name': 'Charlie', 'gender': 'M'}],
    }
    assert groupby(len, []) == {}


def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}
    assert frequencies('mississippi') == {'m': 1, 'i': 4, 's': 4, 'p': 2}
    assert list(frequencies([1, 1, 2]).keys()) == [1, 2]


def test_unique_order_and_key():
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert list(unique('abracadabra')) == ['a', 'b', 'r', 'c', 'd']
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == (
        'cat', 'mouse')


def test_unique_matches_set_of_input():
    data = [3, 1, 3, 2, 1, 5, 2]
    assert set(unique(data)) == set(data)
    assert len(list(unique(data))) == len(set(data))


def test_partition_drops_or_pads_tail():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [
        (1, 2), (3, 4), (5, None)]
    assert list(partition(3, 'abcd', pad='-')) == [
        ('a', 'b', 'c'), ('d', '-', '-')]


def test_partition_all_keeps_short_tail():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(3, iter(range(7)))) == [
        (0, 1, 2), (3, 4, 5), (6,)]
    assert list(partition_all(3, [])) == []


def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, range(5))) == [
        (0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert list(sliding_window(5, [1, 2, 3])) == []
    mean = lambda seq: float(sum(seq)) / len(seq)  # noqa: E731
    assert list(map(mean, sliding_window(2, [1, 2, 3, 4]))) == [1.5, 2.5, 3.5]


def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'
    infinite = interleave([itertools.count(), itertools.count(100)])
    assert list(take(6, infinite)) == [0, 100, 1, 101, 2, 102]


def test_take_and_drop():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(take(10, [1, 2])) == [1, 2]


def test_nth_on_sequence_and_iterator():
    assert nth(1, 'ABC') == 'B'
    assert nth(2, (x * x for x in range(10))) == 4
    with pytest.raises(IndexError):
        nth(5, [1, 2, 3])


def test_first_second_last():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert last('ABC') == 'C'
    assert last(x for x in range(5)) == 4
    with pytest.raises(StopIteration):
        first([])
    with pytest.raises(IndexError):
        last([])


def test_tail_on_list_string_and_iterator():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, 'abcde') == 'de'
    assert tail(2, iter([1, 2, 3])) == (2, 3)


def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]
    assert list(pluck(1, [[1, 2], [3]], default=None)) == [2, None]
    assert list(pluck([0, 2], [[1, 2, 3], [4]], default=0)) == [(1, 3), (4, 0)]


def test_join_inner():
    left = [(1, 'a'), (2, 'b'), (2, 'c')]
    right = [(2, 'x'), (3, 'y')]
    assert list(join(first, left, first, right)) == [
        ((2, 'b'), (2, 'x')), ((2, 'c'), (2, 'x'))]
    people = [('alice', 1), ('bob', 2)]
    ids = [(1, 'x'), (2, 'y'), (2, 'z')]
    assert list(join(1, people, 0, ids)) == [
        (('alice', 1), (1, 'x')),
        (('bob', 2), (2, 'y')),
        (('bob', 2), (2, 'z')),
    ]


def test_join_outer_variants():
    ident = lambda x: x  # noqa: E731
    assert list(join(ident, [1, 2, 3], ident, [2, 3, 4],
                     left_default=None, right_default=None)) == [
        (2, 2), (3, 3), (None, 4), (1, None)]
    assert list(join(ident, [1, 2, 3], ident, [2, 3, 4],
                     left_default=None)) == [(2, 2), (3, 3), (None, 4)]
    assert list(join(ident, [1, 2, 3], ident, [2, 3, 4],
                     right_default=None)) == [(2, 2), (3, 3), (1, None)]


def test_reduceby_sum_and_product_by_parity():
    assert reduceby(iseven, add, [1, 2, 3, 4, 5]) == {False: 9, True: 6}
    assert reduceby(iseven, mul, [1, 2, 3, 4, 5]) == {False: 15, True: 8}


def test_reduceby_with_init():
    projects = [
        {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
        {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
        {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
        {'name': 'help farmers', 'state': 'CA', 'cost': 200000},
    ]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}

    def set_add(s, i):
        s.add(i)
        return s

    sets = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
    assert sets == {True: {2, 4}, False: {1, 3}}


def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == (
        'Charlie', 'Alice')


def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [
        (3, 10), (None, 100)]
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]
    with pytest.raises(TypeError):
        list(diff())


def test_peek_and_peekn_do_not_consume():
    head, seq = peek([0, 1, 2, 3, 4])
    assert head == 0
    assert list(seq) == [0, 1, 2, 3, 4]
    first_two, seq = peekn(2, [0, 1, 2, 3, 4])
    assert first_two == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]
    with pytest.raises(StopIteration):
        peek([])


def test_random_sample_with_seed_is_deterministic():
    seq = list(range(100))
    expected = [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    assert list(random_sample(0.1, seq, random_state=2016)) == expected
    assert list(random_sample(0.1, seq, random_state=2016)) == expected
    assert list(random_sample(0.1, seq, random_state=random.Random(2016))) == expected


def test_random_sample_probabilities_and_subset():
    seq = list(range(200))
    assert list(random_sample(0.0, seq)) == []
    assert list(random_sample(1.0, seq)) == seq
    result = list(random_sample(0.5, seq))
    assert set(result) <= set(seq)
    assert result == sorted(result)
    assert 0 < len(result) < 200


def test_count_concat_mapcat():
    assert count([1, 2, 3]) == 3
    assert count(iter(range(7))) == 7
    assert count(x for x in 'abcd') == 4
    assert count([]) == 0
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat([[1, [2]], [3]])) == [1, [2], 3]
    result = mapcat(lambda s: [c.upper() for c in s],
                    [["a", "b"], ["c", "d", "e"]])
    assert list(result) == ['A', 'B', 'C', 'D', 'E']


def test_isdistinct():
    assert isdistinct([1, 2, 3])
    assert not isdistinct([1, 2, 1])
    assert not isdistinct("Hello")
    assert isdistinct("World")
    assert isdistinct(iter([1, 2, 3]))
    assert not isdistinct(iter([1, 2, 2]))


def test_iterate_and_accumulate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter), next(counter), next(counter)] == [0, 1, 2]
    assert list(take(5, iterate(lambda x: x * 2, 1))) == [1, 2, 4, 8, 16]
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]
    assert list(accumulate(add, [])) == []


def test_remove_filters_out_matches():
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]
    assert list(remove(lambda s: s.startswith('a'),
                       ['apple', 'pear', 'avocado'])) == ['pear']


def test_merge_sorted_and_get():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [
        2, 1, 3, 3]
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(1, 'ABC') == 'B'
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get(5, [1, 2], default=0) == 0
    with pytest.raises(IndexError):
        get(5, [1, 2])
