import pytest

from more_itertools import (
    chunked,
    ichunked,
    sliding_window,
    windowed,
    grouper,
    spy,
    peekable,
    seekable,
    countable,
)


def test_chunked_even_split():
    assert list(chunked([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]


def test_chunked_uneven_last_chunk():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_strict_raises_on_short_chunk():
    with pytest.raises(ValueError):
        list(chunked([1, 2, 3, 4, 5], 2, strict=True))


def test_chunked_strict_ok_when_divisible():
    assert list(chunked("abcd", 2, strict=True)) == [["a", "b"], ["c", "d"]]


def test_ichunked_contents():
    chunks = [list(c) for c in ichunked(range(10), 4)]
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_ichunked_out_of_order_consumption():
    it = ichunked(range(12), 4)
    c1, c2, c3 = next(it), next(it), next(it)
    assert list(c3) == [8, 9, 10, 11]
    assert list(c1) == [0, 1, 2, 3]
    assert list(c2) == [4, 5, 6, 7]


def test_windowed_basic():
    assert list(windowed([1, 2, 3, 4, 5], 3)) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


def test_windowed_fills_short_input():
    assert list(windowed([1, 2], 4, fillvalue="x")) == [(1, 2, "x", "x")]


def test_windowed_step():
    assert list(windowed(range(6), 2, step=2)) == [(0, 1), (2, 3), (4, 5)]


def test_sliding_window_basic():
    assert list(sliding_window("abcd", 2)) == [("a", "b"), ("b", "c"), ("c", "d")]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_sliding_window_number_of_windows(n):
    data = list(range(10))
    assert len(list(sliding_window(data, n))) == len(data) - n + 1


def test_grouper_fill():
    assert list(grouper("ABCDEFG", 3, fillvalue="x")) == [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G", "x", "x"),
    ]


def test_grouper_strict_raises():
    with pytest.raises(ValueError):
        list(grouper("ABCDEFG", 3, incomplete="strict"))


def test_grouper_ignore():
    assert list(grouper("ABCDEFG", 3, incomplete="ignore")) == [
        ("A", "B", "C"),
        ("D", "E", "F"),
    ]


def test_spy_does_not_consume():
    head, it = spy([1, 2, 3, 4], 2)
    assert head == [1, 2]
    assert list(it) == [1, 2, 3, 4]


def test_peekable_peek_and_next():
    p = peekable([1, 2, 3])
    assert p.peek() == 1
    assert p.peek() == 1
    assert next(p) == 1
    assert p.peek() == 2


def test_peekable_peek_default_when_exhausted():
    p = peekable([])
    assert p.peek("empty") == "empty"
    with pytest.raises(StopIteration):
        p.peek()


def test_peekable_prepend():
    p = peekable([1, 2, 3])
    p.prepend(10, 11)
    assert next(p) == 10
    assert list(p) == [11, 1, 2, 3]


def test_seekable_seek_back_to_start():
    s = seekable(str(i) for i in range(5))
    assert [next(s) for _ in range(3)] == ["0", "1", "2"]
    s.seek(0)
    assert list(s) == ["0", "1", "2", "3", "4"]


def test_seekable_seek_past_end():
    s = seekable(range(5))
    s.seek(100)
    assert list(s) == []


def test_seekable_elements_and_relative_seek():
    s = seekable(range(10))
    for _ in range(4):
        next(s)
    s.relative_seek(-2)
    assert next(s) == 2
    assert list(s.elements())[:4] == [0, 1, 2, 3]


def test_countable_counts_items():
    it = countable(map(str, range(10)))
    assert it.items_seen == 0
    next(it)
    next(it)
    assert it.items_seen == 2
    list(it)
    assert it.items_seen == 10
