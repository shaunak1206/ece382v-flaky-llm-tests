import pytest

import more_itertools as mi


def test_one_returns_single_item():
    assert mi.one([7]) == 7
    assert mi.one(iter("x")) == "x"


def test_one_empty_raises_value_error():
    with pytest.raises(ValueError):
        mi.one([])


def test_one_too_long_raises_value_error():
    with pytest.raises(ValueError):
        mi.one([1, 2])


def test_one_custom_exceptions():
    with pytest.raises(IndexError):
        mi.one([], too_short=IndexError("too few items"))
    with pytest.raises(TypeError):
        mi.one([1, 2], too_long=TypeError)


def test_only_single_empty_and_default():
    assert mi.only([1]) == 1
    assert mi.only([], default="missing") == "missing"
    assert mi.only([]) is None


def test_only_too_long_raises():
    with pytest.raises(ValueError):
        mi.only([1, 2])
    with pytest.raises(TypeError):
        mi.only([1, 2], too_long=TypeError)


def test_strictly_n_returns_items_when_length_matches():
    assert list(mi.strictly_n(["a", "b", "c", "d"], 4)) == ["a", "b", "c", "d"]


def test_strictly_n_too_short_raises():
    with pytest.raises(ValueError):
        list(mi.strictly_n("ab", 3))


def test_strictly_n_too_long_raises():
    with pytest.raises(ValueError):
        list(mi.strictly_n("abcd", 3))


def test_spy_default_peeks_one_item_without_consuming():
    head, iterable = mi.spy("abcdefg")
    assert head == ["a"]
    assert list(iterable) == ["a", "b", "c", "d", "e", "f", "g"]


def test_spy_n_larger_than_iterable():
    head, iterable = mi.spy([1, 2], 5)
    assert head == [1, 2]
    assert list(iterable) == [1, 2]


def test_sliced_even_and_uneven():
    assert list(mi.sliced((1, 2, 3, 4, 5, 6), 3)) == [(1, 2, 3), (4, 5, 6)]
    assert list(mi.sliced((1, 2, 3, 4, 5, 6, 7, 8), 3)) == [(1, 2, 3), (4, 5, 6), (7, 8)]


def test_sliced_strict_raises_when_not_divisible():
    with pytest.raises(ValueError):
        list(mi.sliced((1, 2, 3, 4, 5), 3, strict=True))


def test_consumer_advances_to_first_yield():
    received = []

    @mi.consumer
    def collector():
        while True:
            received.append((yield))

    c = collector()
    c.send("a")
    c.send("b")
    assert received == ["a", "b"]
