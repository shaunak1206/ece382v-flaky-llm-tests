"""LLM-generated unit tests for toolz.itertoolz, based on source/docstrings only."""
import pytest

from toolz.itertoolz import (
    groupby, frequencies, unique, partition, partition_all, sliding_window,
    interleave, take, drop, nth, first, second, last, pluck, join, reduceby,
    topk, diff, peek, peekn, random_sample, count, concat, mapcat, isdistinct,
    iterate, accumulate, remove, tail, isiterable, cons, interpose, get,
    concatv, take_nth,
)


def test_groupby_by_len():
    names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    result = groupby(len, names)
    assert result == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_by_member_key():
    data = [{'name': 'Alice', 'gender': 'F'},
            {'name': 'Bob', 'gender': 'M'},
            {'name': 'Charlie', 'gender': 'M'}]
    result = groupby('gender', data)
    assert set(result.keys()) == {'F', 'M'}
    assert len(result['M']) == 2
    assert result['F'][0]['name'] == 'Alice'


def test_frequencies():
    result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
    assert result == {'cat': 3, 'ox': 1, 'pig': 2}
    assert isinstance(result, dict)


def test_unique_no_key():
    assert list(unique([1, 2, 1, 3])) == [1, 2, 3]


def test_unique_with_key():
    result = list(unique(['cat', 'mouse', 'dog', 'hen'], key=len))
    assert result == ['cat', 'mouse']


def test_partition_exact_and_dropped_remainder():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]


def test_partition_with_pad():
    result = list(partition(2, [1, 2, 3, 4, 5], pad=None))
    assert result == [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_sliding_window():
    result = list(sliding_window(2, [1, 2, 3, 4]))
    assert result == [(1, 2), (2, 3), (3, 4)]


def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_take_drop_take_nth():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_tail_on_list_and_lazy_generator():
    # for a list, tail slices and preserves the list type
    assert tail(2, [10, 20, 30, 40, 50]) == [40, 50]
    # for a non-sliceable lazy generator, tail falls back to a deque -> tuple
    result = tail(2, (x for x in [10, 20, 30, 40, 50]))
    assert result == (40, 50)


def test_nth():
    assert nth(1, 'ABC') == 'B'
    assert nth(0, [10, 20, 30]) == 10
    assert nth(2, iter([10, 20, 30, 40])) == 30


def test_first_second_last():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert last('ABC') == 'C'


def test_pluck_single_key_and_multiple_indices():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    result = list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]]))
    assert result == [(1, 2), (4, 5)]


def test_pluck_with_default():
    data = [{'a': 1}, {'b': 2}]
    result = list(pluck('a', data, default=None))
    assert result == [1, None]


def test_join_inner_and_full_outer():
    friends = [('Alice', 'Edith'), ('Zhao', 'Alice')]
    cities = [('Alice', 'NYC'), ('Edith', 'Paris'), ('Zhao', 'Shanghai')]
    result = sorted(join(second, friends, first, cities))
    expected = sorted([
        (('Alice', 'Edith'), ('Edith', 'Paris')),
        (('Zhao', 'Alice'), ('Alice', 'NYC')),
    ])
    assert result == expected

    # Note: result contains None, so it cannot be sorted with sorted();
    # compare against the exact (deterministic) insertion order instead.
    identity = lambda x: x
    full = list(join(identity, [1, 2, 3], identity, [2, 3, 4],
                      left_default=None, right_default=None))
    assert full == [(2, 2), (3, 3), (None, 4), (1, None)]


def test_reduceby_simple_and_with_init():
    from operator import add
    iseven = lambda x: x % 2 == 0
    data = [1, 2, 3, 4, 5]
    assert reduceby(iseven, add, data) == {False: 9, True: 6}

    projects = [{'name': 'a', 'state': 'CA', 'cost': 100},
                {'name': 'b', 'state': 'IL', 'cost': 10},
                {'name': 'c', 'state': 'CA', 'cost': 20}]
    result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
    assert result == {'CA': 120, 'IL': 10}


def test_topk_and_with_key():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    result = topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
    assert result == ('Charlie', 'Alice')


def test_diff_basic():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]


def test_diff_with_default_and_key():
    result = list(diff([1, 2, 3], [1, 2, 10, 100], default=None))
    assert result == [(3, 10), (None, 100)]
    result2 = list(diff(['apples', 'bananas'], ['Apples', 'Oranges'],
                         key=str.lower))
    assert result2 == [('bananas', 'Oranges')]


def test_diff_too_few_sequences_raises():
    with pytest.raises(TypeError):
        list(diff([1, 2, 3]))


def test_peek_and_peekn():
    seq = [0, 1, 2, 3, 4]
    first_val, seq2 = peek(seq)
    assert first_val == 0
    assert list(seq2) == [0, 1, 2, 3, 4]

    first_two, seq3 = peekn(2, seq)
    assert first_two == (0, 1)
    assert list(seq3) == [0, 1, 2, 3, 4]


def test_random_sample_deterministic_seed():
    seq = list(range(100))
    result1 = list(random_sample(0.1, seq, random_state=2016))
    result2 = list(random_sample(0.1, seq, random_state=2016))
    assert result1 == result2
    assert all(item in seq for item in result1)


def test_count():
    assert count([1, 2, 3]) == 3
    assert count(iter([1, 2, 3, 4])) == 4
    assert count([]) == 0


def test_concat_concatv_and_mapcat():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concatv([], ['a'], ['b', 'c'])) == ['a', 'b', 'c']
    result = list(mapcat(lambda s: [c.upper() for c in s],
                          [['a', 'b'], ['c', 'd', 'e']]))
    assert result == ['A', 'B', 'C', 'D', 'E']


def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(4)] == [0, 1, 2, 3]


def test_accumulate_add_and_with_initial():
    from operator import add
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]


def test_remove():
    iseven = lambda x: x % 2 == 0
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


def test_cons_and_interpose():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(interpose('a', [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_get_single_list_and_default():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')
    result = get(['Alice', 'Dennis'], phonebook, None)
    assert result == ('555-1234', None)


def test_get_missing_no_default_raises():
    with pytest.raises(KeyError):
        get('Zeb', {'Alice': 1})
