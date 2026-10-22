from itertools import islice

import pytest

import more_itertools as mi


def test_chunked_even():
    assert list(mi.chunked([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]


def test_chunked_uneven_last_chunk_is_short():
    assert list(mi.chunked([1, 2, 3, 4, 5, 6, 7, 8], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8]]


def test_chunked_strict_raises_when_not_divisible():
    with pytest.raises(ValueError):
        list(mi.chunked([1, 2, 3, 4, 5, 6, 7], 3, strict=True))


def test_first_returns_first_item_or_default():
    assert mi.first([0, 1, 2, 3]) == 0
    assert mi.first([], "some default") == "some default"


def test_first_empty_without_default_raises():
    with pytest.raises(ValueError):
        mi.first([])


def test_last_returns_last_item_or_default():
    assert mi.last([0, 1, 2, 3]) == 3
    assert mi.last([], "some default") == "some default"


def test_last_empty_without_default_raises():
    with pytest.raises(ValueError):
        mi.last([])


def test_nth_or_last_within_range():
    assert mi.nth_or_last([0, 1, 2, 3], 2) == 2


def test_nth_or_last_beyond_end_returns_last():
    assert mi.nth_or_last([0, 1], 2) == 1


def test_nth_or_last_empty_default_and_error():
    assert mi.nth_or_last([], 0, "some default") == "some default"
    with pytest.raises(ValueError):
        mi.nth_or_last([], 0)


def test_peekable_peek_does_not_advance():
    p = mi.peekable(["a", "b"])
    assert p.peek() == "a"
    assert p.peek() == "a"
    assert next(p) == "a"
    assert next(p) == "b"


def test_peekable_peek_default_when_exhausted():
    p = mi.peekable([])
    assert p.peek("hi") == "hi"


def test_peekable_peek_empty_raises_stopiteration():
    p = mi.peekable([])
    with pytest.raises(StopIteration):
        p.peek()


def test_peekable_prepend():
    p = mi.peekable([1, 2, 3])
    p.prepend(10, 11, 12)
    assert next(p) == 10
    assert p.peek() == 11
    assert list(p) == [11, 12, 1, 2, 3]


def test_ilen_counts_items():
    assert mi.ilen(x for x in range(1000) if x % 2 == 0) == 500
    assert mi.ilen([]) == 0


def test_iterate_produces_repeated_applications():
    result = list(islice(mi.iterate(lambda x: 2 * x, 1), 10))
    assert result == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
