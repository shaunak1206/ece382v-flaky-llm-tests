from itertools import count

import pytest

from more_itertools import peekable, seekable


# ---------------------------------------------------------------- peekable


def test_peekable_peek_does_not_advance():
    p = peekable(['a', 'b'])
    assert p.peek() == 'a'
    assert p.peek() == 'a'
    assert next(p) == 'a'
    assert p.peek() == 'b'
    assert list(p) == ['b']


def test_peekable_peek_default_and_exhausted():
    p = peekable([])
    assert p.peek('hi') == 'hi'
    with pytest.raises(StopIteration):
        p.peek()
    with pytest.raises(StopIteration):
        next(p)


def test_peekable_prepend():
    p = peekable([1, 2, 3])
    p.prepend(10, 11, 12)
    assert next(p) == 10
    assert p.peek() == 11
    assert list(p) == [11, 12, 1, 2, 3]

    p = peekable([])
    with pytest.raises(StopIteration):
        next(p)
    p.prepend(1)
    assert next(p) == 1
    with pytest.raises(StopIteration):
        next(p)


def test_peekable_indexing_caches_items():
    p = peekable(['a', 'b', 'c', 'd'])
    assert p[0] == 'a'
    assert p[1] == 'b'
    assert next(p) == 'a'
    assert p[-1] == 'd'
    assert list(p) == ['b', 'c', 'd']


def test_peekable_slicing_and_bool():
    p = peekable(range(5))
    assert p[1:3] == [1, 2]
    assert list(p) == [0, 1, 2, 3, 4]
    assert not p
    q = peekable(['a', 'b'])
    assert q
    assert list(q) == ['a', 'b']
    assert not q


# ---------------------------------------------------------------- seekable


def test_seekable_seek_back_to_start():
    it = seekable(str(n) for n in count())
    assert (next(it), next(it), next(it)) == ('0', '1', '2')
    it.seek(0)
    assert (next(it), next(it), next(it)) == ('0', '1', '2')
    assert next(it) == '3'


def test_seekable_seek_forward_and_past_end():
    it = seekable(str(n) for n in range(20))
    it.seek(10)
    assert next(it) == '10'
    it.seek(20)
    assert list(it) == []
    it.seek(0)
    assert next(it) == '0'


def test_seekable_relative_seek():
    it = seekable(str(n) for n in range(20))
    assert (next(it), next(it), next(it)) == ('0', '1', '2')
    it.relative_seek(2)
    assert next(it) == '5'
    it.relative_seek(-3)
    assert next(it) == '3'
    it.relative_seek(-3)
    assert next(it) == '1'


def test_seekable_peek_bool_and_getitem():
    it = seekable('1234')
    assert it.peek() == '1'
    assert list(it) == ['1', '2', '3', '4']
    assert it.peek(default='empty') == 'empty'
    assert not it

    other = seekable('5678')
    assert bool(other)
    assert list(other) == ['5', '6', '7', '8']
    assert not bool(other)

    third = seekable(str(n) for n in range(10))
    next(third), next(third), next(third)
    assert third[-1] == '2'
    assert third[0] == '0'


def test_seekable_elements_view_and_maxlen():
    it = seekable(str(n) for n in range(10))
    next(it), next(it), next(it)
    elements = it.elements()
    assert list(elements) == ['0', '1', '2']
    assert next(it) == '3'
    assert list(elements) == ['0', '1', '2', '3']

    limited = seekable((str(n) for n in count()), maxlen=2)
    assert (next(limited), next(limited), next(limited), next(limited)) == (
        '0', '1', '2', '3')
    assert list(limited.elements()) == ['2', '3']
    limited.seek(0)
    assert (next(limited), next(limited), next(limited), next(limited)) == (
        '2', '3', '4', '5')
    assert next(limited) == '6'
