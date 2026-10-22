from itertools import count
from operator import add

import pytest

from toolz import curry, unique
from toolz.itertoolz import take
from toolz.sandbox.core import EqualityHashKey, unzip
from toolz.sandbox.parallel import fold


def test_equality_hash_key_makes_unhashables_hashable():
    default_key = curry(EqualityHashKey, None)
    items = [[], (), [1], [1]]
    assert len(set(map(default_key, items))) == 3

    by_len = curry(EqualityHashKey, len)
    assert by_len([1]) == by_len([1])
    assert by_len([1]) != by_len([2])
    assert len(set(map(by_len, items))) == 3

    vals = [[], [], (), [1], [1], [2], {}, {}, {}]
    assert list(unique(vals, key=default_key)) == [[], (), [1], [2], {}]

    key = EqualityHashKey(None, [1, 2])
    assert repr(key) == '=[1, 2]='
    assert str(key) == '=[1, 2]='
    assert key != 5
    assert EqualityHashKey(0, [3, 4]) == EqualityHashKey(0, [3, 4])


def test_unzip_inverts_zip():
    a, b = unzip([('a', 1), ('b', 2)])
    assert list(a) == ['a', 'b']
    assert list(b) == [1, 2]

    x, y, z = unzip([(1, 2, 3), (4, 5, 6)])
    assert (list(x), list(y), list(z)) == ([1, 4], [2, 5], [3, 6])

    assert unzip([]) == ()

    # works on an infinite outer sequence
    left, right = unzip(zip(count(0), count(10)))
    assert list(take(3, left)) == [0, 1, 2]
    assert list(take(3, right)) == [10, 11, 12]


def test_fold_reduces_in_chunks():
    assert fold(add, [1, 2, 3, 4], chunksize=2) == 10
    assert fold(add, [1, 2, 3], chunksize=10) == 6
    assert fold(add, range(1000), chunksize=10) == sum(range(1000))
    assert fold(add, range(1, 11), default=0, chunksize=3) == 55

    # binop and combine have different types: (total, item) vs (total, total)
    assert fold(lambda total, s: total + len(s), ['a', 'bb', 'ccc', 'dddd'],
                default=0, chunksize=2, combine=add) == 10

    calls = []

    def recording_map(func, chunks):
        chunks = list(chunks)
        calls.append(len(chunks))
        return [func(chunk) for chunk in chunks]

    assert fold(add, list(range(8)), chunksize=2, map=recording_map) == 28
    assert calls[0] == 4

    with pytest.raises(AssertionError):
        fold(add, [1, 2, 3], chunksize=1)
