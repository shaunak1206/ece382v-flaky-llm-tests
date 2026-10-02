import itertools

import pytest

from more_itertools import (
    collapse,
    interleave_longest,
    partition,
    roundrobin,
    run_length,
    sliding_window,
    split_at,
    split_before,
    windowed,
)


def test_windowed_basic_and_padding():
    assert list(windowed([1, 2, 3, 4, 5], 3)) == [
        (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    assert list(windowed([1, 2, 3], 4)) == [(1, 2, 3, None)]
    assert list(windowed([1, 2, 3], 5, fillvalue='-')) == [
        (1, 2, 3, '-', '-')]
    assert list(windowed([], 3)) == []


def test_windowed_with_step():
    assert list(windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2)) == [
        (1, 2, 3), (3, 4, 5), (5, 6, '!')]
    assert list(windowed([1, 2, 3, 4, 5], 3, step=2)) == [
        (1, 2, 3), (3, 4, 5)]


def test_windowed_invalid_arguments_and_sliding_in():
    with pytest.raises(ValueError):
        list(windowed([1, 2, 3], 0))
    with pytest.raises(ValueError):
        list(windowed([1, 2, 3], 2, step=0))
    padded = itertools.chain([None] * 2, [1, 2, 3, 4])
    assert list(windowed(padded, 3)) == [
        (None, None, 1), (None, 1, 2), (1, 2, 3), (2, 3, 4)]


def test_sliding_window_sizes():
    assert list(sliding_window(range(6), 4)) == [
        (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
    assert list(sliding_window('abc', 2)) == [('a', 'b'), ('b', 'c')]
    assert list(sliding_window('abc', 1)) == [('a',), ('b',), ('c',)]
    assert list(sliding_window(range(3), 4)) == []


def test_sliding_window_large_window_and_invalid():
    windows = list(sliding_window(range(25), 21))
    assert len(windows) == 5
    assert all(len(w) == 21 for w in windows)
    assert windows[0][0] == 0 and windows[-1][-1] == 24
    with pytest.raises(ValueError):
        sliding_window(range(5), 0)


def test_split_at_basic_and_options():
    assert list(split_at('abcdcba', lambda x: x == 'b')) == [
        ['a'], ['c', 'd', 'c'], ['a']]
    assert list(split_at(range(10), lambda n: n % 2 == 1)) == [
        [0], [2], [4], [6], [8], []]
    assert list(split_at(range(10), lambda n: n % 2 == 1, maxsplit=2)) == [
        [0], [2], [4, 5, 6, 7, 8, 9]]
    assert list(split_at('abcdcba', lambda x: x == 'b',
                         keep_separator=True)) == [
        ['a'], ['b'], ['c', 'd', 'c'], ['b'], ['a']]


def test_split_at_degenerate_inputs():
    assert list(split_at([1, 2, 3], lambda x: x == 2, maxsplit=0)) == [
        [1, 2, 3]]
    assert list(split_at([1, 2, 3], lambda x: False)) == [[1, 2, 3]]
    assert list(split_at([], lambda x: True)) == [[]]


def test_split_before():
    assert list(split_before('OneTwo', lambda s: s.isupper())) == [
        ['O', 'n', 'e'], ['T', 'w', 'o']]
    assert list(split_before(range(10), lambda n: n % 3 == 0)) == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert list(split_before(range(10), lambda n: n % 3 == 0,
                             maxsplit=2)) == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    assert list(split_before([], lambda x: True)) == []


def test_collapse_nested_iterables_and_strings():
    assert list(collapse([(1, 2), ([3, 4], [[5], [6]])])) == [
        1, 2, 3, 4, 5, 6]
    assert list(collapse(['ab', ['cd', ('ef',)], b'gh'])) == [
        'ab', 'cd', 'ef', b'gh']
    assert list(collapse(5)) == [5]


def test_collapse_base_type_and_levels():
    iterable = ['ab', ('cd', 'ef'), ['gh', 'ij']]
    assert list(collapse(iterable, base_type=tuple)) == [
        'ab', ('cd', 'ef'), 'gh', 'ij']
    nested = [('a', ['b']), ('c', ['d'])]
    assert list(collapse(nested)) == ['a', 'b', 'c', 'd']
    assert list(collapse(nested, levels=1)) == ['a', ['b'], 'c', ['d']]


def test_run_length_encode_decode_roundtrip():
    assert list(run_length.encode('abbcccdddd')) == [
        ('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    assert list(run_length.encode([])) == []
    compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    assert list(run_length.decode(compressed)) == [
        'a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']
    data = list('aaabccdddd')
    assert list(run_length.decode(run_length.encode(data))) == data


def test_roundrobin_and_interleave_longest():
    assert list(roundrobin('ABC', 'D', 'EF')) == ['A', 'D', 'E', 'B', 'F', 'C']
    assert list(roundrobin()) == []
    assert list(interleave_longest([1, 2, 3], [4, 5], [6, 7, 8])) == [
        1, 4, 6, 2, 5, 7, 3, 8]
    assert list(interleave_longest()) == []
    assert list(interleave_longest('ab', '')) == ['a', 'b']
    iterables = ([1, 2, 3], [4], [5, 6], [])
    assert list(roundrobin(*iterables)) == list(interleave_longest(*iterables))


def test_partition_by_predicate():
    is_odd = lambda x: x % 2 != 0  # noqa: E731
    evens, odds = partition(is_odd, range(10))
    assert list(evens) == [0, 2, 4, 6, 8]
    assert list(odds) == [1, 3, 5, 7, 9]
    falsy, truthy = partition(None, [0, 1, False, True, '', ' '])
    assert list(falsy) == [0, False, '']
    assert list(truthy) == [1, True, ' ']


def test_partition_consumed_in_either_order():
    evens, odds = partition(lambda x: x % 2, range(6))
    assert list(odds) == [1, 3, 5]
    assert list(evens) == [0, 2, 4]
