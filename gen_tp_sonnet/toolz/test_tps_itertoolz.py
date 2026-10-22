from operator import add, mul

import pytest

import toolz
from toolz import (
    accumulate, concat, concatv, cons, count, diff, drop, first, frequencies,
    get, groupby, interleave, interpose, isdistinct, isiterable, iterate,
    join, last, mapcat, merge_sorted, nth, partition, partition_all, peek,
    peekn, pluck, random_sample, reduceby, remove, second, sliding_window,
    tail, take, take_nth, topk, unique,
)


def iseven(x):
    return x % 2 == 0


def test_remove_basic():
    assert list(remove(iseven, [1, 2, 3, 4])) == [1, 3]


def test_accumulate_add_and_mul():
    assert list(accumulate(add, [1, 2, 3, 4, 5])) == [1, 3, 6, 10, 15]
    assert list(accumulate(mul, [1, 2, 3, 4, 5])) == [1, 2, 6, 24, 120]


def test_accumulate_with_initial():
    assert list(accumulate(add, [1, 2, 3], -1)) == [-1, 0, 2, 5]
    assert list(accumulate(add, [], 1)) == [1]


def test_groupby_callable_key():
    result = groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    assert result == {
        3: ['Bob', 'Dan'],
        5: ['Alice', 'Edith', 'Frank'],
        7: ['Charlie'],
    }


def test_groupby_parity():
    assert groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8]) == {
        False: [1, 3, 5, 7],
        True: [2, 4, 6, 8],
    }


def test_merge_sorted_basic():
    assert list(merge_sorted([1, 3, 5], [2, 4, 6])) == [1, 2, 3, 4, 5, 6]
    assert ''.join(merge_sorted('abc', 'abc', 'abc')) == 'aaabbbccc'


def test_merge_sorted_key():
    assert list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3)) == [2, 1, 3, 3]


def test_interleave():
    assert list(interleave([[1, 2], [3, 4]])) == [1, 3, 2, 4]
    assert ''.join(interleave(('ABC', 'XY'))) == 'AXBYC'


def test_unique_and_key():
    assert tuple(unique((1, 2, 3))) == (1, 2, 3)
    assert tuple(unique((1, 2, 1, 3))) == (1, 2, 3)
    assert tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len)) == ('cat', 'mouse')


def test_isiterable():
    assert isiterable([1, 2, 3]) is True
    assert isiterable('abc') is True
    assert isiterable(5) is False


def test_isdistinct():
    assert isdistinct([1, 2, 3]) is True
    assert isdistinct([1, 2, 1]) is False
    assert isdistinct("Hello") is False
    assert isdistinct("World") is True


def test_take_and_drop():
    assert list(take(2, [10, 20, 30, 40, 50])) == [10, 20]
    assert list(drop(2, [10, 20, 30, 40, 50])) == [30, 40, 50]


def test_tail():
    assert list(tail(2, [10, 20, 30, 40, 50])) == [40, 50]


def test_take_nth():
    assert list(take_nth(2, [10, 20, 30, 40, 50])) == [10, 30, 50]


def test_first_second_nth_last():
    assert first('ABC') == 'A'
    assert second('ABC') == 'B'
    assert nth(1, 'ABC') == 'B'
    assert last('ABC') == 'C'


def test_get_single_and_list():
    assert get(1, 'ABC') == 'B'
    assert get([1, 2], 'ABC') == ('B', 'C')


def test_get_dict_and_defaults():
    phonebook = {'Alice': '555-1234', 'Bob': '555-5678', 'Charlie': '555-9999'}
    assert get('Alice', phonebook) == '555-1234'
    assert get(['Alice', 'Bob'], phonebook) == ('555-1234', '555-5678')
    assert get(['Alice', 'Dennis'], phonebook, None) == ('555-1234', None)


def test_get_missing_without_default_raises():
    with pytest.raises(KeyError):
        get('nope', {'a': 1})
    with pytest.raises(IndexError):
        get(10, [1, 2, 3])


def test_concat_and_concatv():
    assert list(concat([[], [1], [2, 3]])) == [1, 2, 3]
    assert list(concatv([], ["a"], ["b", "c"])) == ['a', 'b', 'c']


def test_mapcat():
    result = mapcat(lambda s: [c.upper() for c in s], [["a", "b"], ["c", "d", "e"]])
    assert list(result) == ['A', 'B', 'C', 'D', 'E']


def test_cons_and_interpose():
    assert list(cons(1, [2, 3])) == [1, 2, 3]
    assert list(interpose("a", [1, 2, 3])) == [1, 'a', 2, 'a', 3]


def test_frequencies():
    assert frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat']) == {
        'cat': 3, 'ox': 1, 'pig': 2}


def test_reduceby():
    result = reduceby(iseven, add, [1, 2, 3, 4, 5], 0)
    assert result == {False: 9, True: 6}


def test_iterate():
    counter = iterate(lambda x: x + 1, 0)
    assert [next(counter) for _ in range(3)] == [0, 1, 2]
    powers = iterate(lambda x: x * 2, 1)
    assert [next(powers) for _ in range(4)] == [1, 2, 4, 8]


def test_sliding_window():
    assert list(sliding_window(2, [1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]


def test_partition():
    assert list(partition(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4)]
    assert list(partition(2, [1, 2, 3, 4, 5], pad=None)) == [(1, 2), (3, 4), (5, None)]


def test_partition_all():
    assert list(partition_all(2, [1, 2, 3, 4])) == [(1, 2), (3, 4)]
    assert list(partition_all(2, [1, 2, 3, 4, 5])) == [(1, 2), (3, 4), (5,)]


def test_pluck():
    data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
    assert list(pluck('name', data)) == ['Cheese', 'Pies']
    assert list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]])) == [(1, 2), (4, 5)]


def test_join_inner():
    left = [(1, 'a'), (2, 'b')]
    right = [(1, 'x'), (3, 'y')]
    assert list(join(0, left, 0, right)) == [((1, 'a'), (1, 'x'))]


def test_diff():
    assert list(diff([1, 2, 3], [1, 2, 10, 100])) == [(3, 10)]
    assert list(diff([1, 2, 3], [1, 2, 10, 100], default=None)) == [(3, 10), (None, 100)]
    assert list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower)) == [
        ('bananas', 'Oranges')]


def test_topk():
    assert topk(2, [1, 100, 10, 1000]) == (1000, 100)
    assert topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len) == ('Charlie', 'Alice')


def test_peek_preserves_sequence():
    head, seq = peek([0, 1, 2, 3, 4])
    assert head == 0
    assert list(seq) == [0, 1, 2, 3, 4]


def test_random_sample_extremes():
    seq = list(range(50))
    assert list(random_sample(0, seq)) == []
    assert list(random_sample(1, seq)) == seq

