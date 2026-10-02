import itertools

import pytest

from more_itertools import (
    always_iterable,
    chunked,
    consume,
    countable,
    first,
    grouper,
    ichunked,
    last,
    nth,
    one,
    only,
    spy,
)


def test_chunked_splits_into_lists():
    assert list(chunked([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]
    assert list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3)) == [
        [1, 2, 3], [4, 5, 6], [7, 8]]
    assert list(chunked([], 3)) == []
    assert list(chunked('ab', 5)) == [['a', 'b']]


def test_chunked_strict_mode():
    assert list(chunked([1, 2, 3, 4], 2, strict=True)) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3, 4, 5], 2, strict=True))


def test_chunked_invalid_arguments_and_infinite_input():
    with pytest.raises(ValueError):
        chunked([1, 2, 3], -1)
    with pytest.raises(ValueError):
        chunked([1, 2, 3], None, strict=True)
    chunks = chunked(itertools.count(), 3)
    assert next(chunks) == [0, 1, 2]
    assert next(chunks) == [3, 4, 5]


def test_first_item_default_and_empty():
    assert first([0, 1, 2, 3]) == 0
    assert first('xyz') == 'x'
    assert first([], 'some default') == 'some default'
    assert first([], None) is None
    with pytest.raises(ValueError):
        first([])


def test_last_item_default_and_empty():
    assert last([0, 1, 2, 3]) == 3
    assert last(x * 2 for x in range(4)) == 6
    assert last('abc') == 'c'
    assert last([], 'some default') == 'some default'
    with pytest.raises(ValueError):
        last([])


def test_one_returns_exactly_one_item():
    assert one([5]) == 5
    assert one(iter('a')) == 'a'
    with pytest.raises(ValueError):
        one([])
    with pytest.raises(ValueError):
        one([1, 2])
    with pytest.raises(IndexError):
        one([], too_short=IndexError('too few items'))
    with pytest.raises(RuntimeError):
        one(['too', 'many'], too_long=RuntimeError)


def test_only_cases():
    assert only([]) is None
    assert only([], default='missing') == 'missing'
    assert only([1]) == 1
    with pytest.raises(ValueError):
        only([1, 2])
    with pytest.raises(TypeError):
        only([1, 2], too_long=TypeError)


def test_nth_item_or_default():
    assert nth(range(10), 3) == 3
    assert nth(range(10), 20, 'zebra') == 'zebra'
    assert nth('abc', 1) == 'b'
    assert nth([], 0) is None


def test_consume_partial_full_and_overrun():
    it = iter(range(10))
    consume(it, 3)
    assert next(it) == 3
    consume(it)
    assert next(it, 'done') == 'done'
    short = iter(range(3))
    consume(short, 5)
    assert list(short) == []


def test_always_iterable():
    assert list(always_iterable(1)) == [1]
    assert list(always_iterable('foo')) == ['foo']
    assert list(always_iterable(b'bar')) == [b'bar']
    assert list(always_iterable(None)) == []
    assert list(always_iterable((1, 2, 3))) == [1, 2, 3]
    assert list(always_iterable({'a': 1})) == ['a']
    assert list(always_iterable({'a': 1}, base_type=dict)) == [{'a': 1}]
    assert list(always_iterable('foo', base_type=None)) == ['f', 'o', 'o']


def test_spy_does_not_consume():
    head, iterable = spy('abcdefg')
    assert head == ['a']
    assert list(iterable) == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    (a, b), iterable = spy('abcdefg', 2)
    assert (a, b) == ('a', 'b')
    assert list(iterable) == list('abcdefg')
    head, iterable = spy([1, 2, 3, 4, 5], 10)
    assert head == [1, 2, 3, 4, 5]
    assert list(iterable) == [1, 2, 3, 4, 5]
    head, iterable = spy([])
    assert head == []
    assert list(iterable) == []


def test_ichunked_in_and_out_of_order():
    assert [list(c) for c in ichunked(range(7), 3)] == [
        [0, 1, 2], [3, 4, 5], [6]]
    chunks = list(ichunked(range(9), 3))
    assert list(chunks[2]) == [6, 7, 8]
    assert list(chunks[0]) == [0, 1, 2]
    assert list(chunks[1]) == [3, 4, 5]


def test_ichunked_infinite_and_invalid():
    all_chunks = ichunked(itertools.count(), 4)
    c_1, c_2 = next(all_chunks), next(all_chunks)
    assert list(c_2) == [4, 5, 6, 7]
    assert list(c_1) == [0, 1, 2, 3]
    assert list(ichunked([1, 2, 3], 0)) == []
    with pytest.raises(ValueError):
        list(ichunked([1, 2, 3], -1))


def test_countable_tracks_items_seen():
    it = countable(map(str, range(10)))
    assert it.items_seen == 0
    assert (next(it), next(it)) == ('0', '1')
    assert it.items_seen == 2
    assert list(it) == ['2', '3', '4', '5', '6', '7', '8', '9']
    assert it.items_seen == 10
    empty = countable([])
    assert list(empty) == []
    assert empty.items_seen == 0


def test_grouper_modes():
    assert list(grouper('ABCDEF', 3)) == [('A', 'B', 'C'), ('D', 'E', 'F')]
    assert list(grouper('ABCDE', 2)) == [('A', 'B'), ('C', 'D'), ('E', None)]
    assert list(grouper('ABCDEFG', 3, incomplete='fill', fillvalue='x')) == [
        ('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    assert list(grouper('ABCDEFG', 3, incomplete='ignore')) == [
        ('A', 'B', 'C'), ('D', 'E', 'F')]
    assert list(grouper('ABCDEF', 3, incomplete='strict')) == [
        ('A', 'B', 'C'), ('D', 'E', 'F')]
    with pytest.raises(ValueError):
        list(grouper('ABCDEFG', 3, incomplete='strict'))
    with pytest.raises(ValueError):
        grouper('ABC', 2, incomplete='bogus')
