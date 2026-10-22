import itertools
from operator import add, mul

import pytest

from toolz.itertoolz import (
    accumulate, concat, concatv, cons, count, diff, drop, first, frequencies,
    get, groupby, interleave, interpose, isdistinct, isiterable, iterate, join,
    last, mapcat, merge_sorted, nth, partition, partition_all, peek, peekn,
    pluck, random_sample, reduceby, remove, second, sliding_window, tail,
    take, take_nth, topk, unique,
)


def iseven(x):
    return x % 2 == 0


def inc(x):
    return x + 1


def test_remove_keeps_items_failing_predicate():
    assert list(remove(iseven, [1, 2, 3, 4, 5])) == [1, 3, 5]
    assert list(remove(iseven, [])) == []


def test_accumulate_without_initial():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]
    assert list(accumulate(add, [])) == []


def test_accumulate_with_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_groupby_with_callable_key():
    result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
    assert result == {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
    assert groupby(len, ['Alice', 'Bob', 'Dan']) == {5: ['Alice'],
                                                      3: ['Bob', 'Dan']}


def test_groupby_with_non_callable_key_groups_on_member():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert result == {'F': [data[0]], 'M': [data[1], data[2]]}
    assert groupby(0, []) == {}


def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'
    assert list(merge_sorted([1, 2], [])) == [1, 2]
    assert list(merge_sorted([], [3, 4])) == [3, 4]


def test_merge_sorted_key_empty_and_single():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == \
        [2, 1, 3, 3]
    assert list(merge_sorted()) == []
    assert list(merge_sorted([1, 2, 3])) == [1, 2, 3]


def test_interleave_finite_and_infinite():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'
    infinite = interleave([itertools.count(0), itertools.count(100)])
    assert list(take(5, infinite)) == [0, 100, 1, 101, 2]


def test_unique_with_and_without_key():
    assert tuple(unique((1, 2, 1, 3, 2))) == (1, 2, 3)
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == \
        ('cat', 'mouse')
    assert tuple(unique([])) == ()


def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False
    assert isiterable(None) is False


def test_isdistinct_on_sequences_and_iterators():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True
    assert isdistinct(iter([1, 2, 3])) is True
    assert isdistinct(iter([1, 2, 2])) is False


def test_take_and_drop():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(take(10, [1, 2])) == [1, 2]
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(drop(10, [1, 2])) == []


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]
    assert list(take_nth(1, 'abc')) == ['a', 'b', 'c']


def test_first_and_second():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert first(iter([7, 8])) == 7
    with pytest.raises(StopIteration):
        first([])
    with pytest.raises(StopIteration):
        second([1])


def test_nth_on_sequences_and_iterators():
    assert nth(1, 'ABC') == 'B'
    assert nth(2, [5, 6, 7, 8]) == 7
    assert nth(2, iter(range(10, 20))) == 12
    with pytest.raises(IndexError):
        nth(5, [1, 2])


def test_tail_and_last():
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    assert tail(2, 'abcde') == 'de'
    assert tail(2, iter([1, 2, 3, 4])) == (3, 4)
    assert last('ABC') == 'C'
    assert last(iter([1, 2, 3])) == 3


def test_get_single_index_and_dict_key():
    assert get(1, 'ABC') == 'B'
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    with pytest.raises(KeyError):
        get('Zed', phonebook)
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_get_with_list_of_indices():
    assert get([1, 2], 'ABC') == ('B', 'C')
    assert get([0], 'ABC') == ('A',)
    assert get([], 'ABC') == ()
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')


def test_get_with_default():
    phonebook = {'Alice': '555-1234'}
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)
    assert get(5, [1, 2], 'missing') == 'missing'
    assert get('x', {}, 0) == 0
    assert get('a', 5, 'dflt') == 'dflt'


def test_concat_concatv_mapcat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concat(([i] * i for i in range(3)))) == [1, 2, 2]
    assert list(concatv([], ['a'], ['b', 'c'])) == ['a', 'b', 'c']
    assert list(mapcat(lambda s: [c.upper() for c in s],
                       [['a', 'b'], ['c', 'd', 'e']])) == \
        ['A', 'B', 'C', 'D', 'E']


def test_cons_and_interpose():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(take(3, cons(0, iterate(inc, 1)))) == [0, 1, 2]
    assert list(interpose('a', [1, 2, 3])) == [1, 'a', 2, 'a', 3]
    assert list(interpose('-', [9])) == [9]


def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == \
        {'cat': 3, 'ox': 1, 'pig': 2}
    assert frequencies([]) == {}
    assert frequencies('aab') == {'a': 2, 'b': 1}


def test_reduceby_variants():
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}
    assert reduceby(iseven, mul, data) == {False: 15, True: 8}
    assert reduceby(iseven, add, data, 100) == {False: 109, True: 106}

    def set_add(s, i):
        s.add(i)
        return s

    assert reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set) == \
        {True: {2, 4}, False: {1, 3}}


def test_reduceby_with_string_key():
    projects = [{'state': 'CA', 'cost': 1000000},
                {'state': 'IL', 'cost': 100000},
                {'state': 'IL', 'cost': 2000000},
                {'state': 'CA', 'cost': 200000}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 1200000, 'IL': 2100000}


def test_iterate_is_lazy_and_repeats_function():
    assert list(take(5, iterate(lambda x: x * 2, 1))) == [1, 2, 4, 8, 16]
    counter = iterate(inc, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]


def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
    assert list(sliding_window(3, 'abcde')) == [('a', 'b', 'c'),
                                                ('b', 'c', 'd'),
                                                ('c', 'd', 'e')]
    assert list(sliding_window(5, [1, 2])) == []


def test_partition_drops_or_pads_remainder():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == \
        [(1, 2), (3, 4), (5, None)]


def test_partition_all_keeps_short_final_chunk():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]
    assert list(partition_all(3, [1, 2, 3, 4, 5, 6, 7])) == \
        [(1, 2, 3), (4, 5, 6), (7,)]
    assert list(partition_all(3, [])) == []


def test_partition_all_on_iterator_without_len():
    assert list(partition_all(2, iter(range(5)))) == [(0, 1), (2, 3), (4,)]
    assert list(partition_all(4, iter(range(6)))) == [(0, 1, 2, 3), (4, 5)]
    assert list(partition_all(3, iter(range(3)))) == [(0, 1, 2)]


def test_count_on_sized_and_lazy_sequences():
    assert count([1, 2, 3]) == 3
    assert count('hello') == 5
    assert count(x for x in range(7)) == 7
    assert count(iter([])) == 0


def test_pluck_single_and_multiple():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]
    assert list(pluck([0], [[1, 2], [3, 4]])) == [(1,), (3,)]
    assert list(pluck([], [[1, 2], [3, 4]])) == [(), ()]


def test_pluck_with_default():
    data = [[1, 2, 3], [4]]
    assert list(pluck(2, data, default=None)) == [3, None]
    assert list(pluck([0, 2], data, default=0)) == [(1, 3), (4, 0)]
    assert list(pluck('k', [{'k': 1}, {}], default='x')) == [1, 'x']


def test_join_inner():
    friends = [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
               ('Zhao', 'Alice'), ('Zhao', 'Edith')]
    cities = [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Sydney'),
              ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    result = join(second, friends, first, cities)
    pairs = {(a, d) for ((a, b), (c, d)) in result}
    assert pairs == {('Alice', 'Berlin'), ('Alice', 'Paris'),
                     ('Alice', 'Shanghai'), ('Edith', 'Chicago'),
                     ('Edith', 'NYC'), ('Zhao', 'Chicago'), ('Zhao', 'NYC'),
                     ('Zhao', 'Berlin'), ('Zhao', 'Paris')}


def test_join_outer_variants():
    def ident(x):
        return x

    left, right = [1, 2, 3], [2, 3, 4]
    assert list(join(ident, left, ident, right)) == [(2, 2), (3, 3)]
    assert list(join(ident, left, ident, right,
                     left_default=None, right_default=None)) == \
        [(2, 2), (3, 3), (None, 4), (1, None)]
    assert list(join(ident, left, ident, right, right_default=None)) == \
        [(2, 2), (3, 3), (1, None)]
    assert list(join(ident, left, ident, right, left_default=None)) == \
        [(2, 2), (3, 3), (None, 4)]


def test_join_with_index_keys_and_duplicate_matches():
    left = [(1, 'a'), (2, 'b'), (1, 'c')]
    right = [('x', 1), ('y', 3)]
    result = list(join(0, left, 1, right))
    assert result == [((1, 'a'), ('x', 1)), ((1, 'c'), ('x', 1))]


def test_diff_basic_default_and_key():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == \
        [(3, 10), (None, 100)]
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                     key=str.lower)) == [('bananas', 'Oranges')]
    assert list(diff([[1, 2], [1, 3]])) == [(2, 3)]


def test_diff_requires_two_sequences():
    with pytest.raises(TypeError):
        list(diff((1, 2)))
    with pytest.raises(TypeError):
        list(diff())


def test_topk_with_and_without_key():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == \
        ('Charlie', 'Alice')
    assert topk(2, [(1, 'a'), (3, 'b'), (2, 'c')], key=0) == \
        ((3, 'b'), (2, 'c'))
    assert topk(5, [3, 1, 2]) == (3, 2, 1)


def test_peek_and_peekn_preserve_sequence():
    head, seq = peek([0, 1, 2, 3, 4])
    assert head == 0
    assert list(seq) == [0, 1, 2, 3, 4]

    firsts, seq = peekn(2, iter([0, 1, 2, 3, 4]))
    assert firsts == (0, 1)
    assert list(seq) == [0, 1, 2, 3, 4]

    with pytest.raises(StopIteration):
        peek([])


def test_random_sample_is_deterministic_for_a_seed():
    seq = list(range(100))
    a = list(random_sample(0.3, seq, random_state=2016))
    b = list(random_sample(0.3, seq, random_state=2016))
    assert a == b
    assert set(a) <= set(seq)
    assert a == sorted(a)


def test_random_sample_extreme_probabilities_and_custom_state():
    seq = list(range(20))
    assert list(random_sample(0, seq, random_state=1)) == []
    assert list(random_sample(1, seq, random_state=1)) == seq

    class Fixed:
        def __init__(self):
            self.values = itertools.cycle([0.1, 0.9])

        def random(self):
            return next(self.values)

    assert list(random_sample(0.5, [10, 20, 30, 40], Fixed())) == [10, 30]
