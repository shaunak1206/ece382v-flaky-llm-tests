from operator import add

from toolz import countby, curry, partitionby
from toolz.sandbox.core import EqualityHashKey, unzip
from toolz.sandbox.parallel import fold


def test_countby_len():
    assert countby(len, ['cat', 'mouse', 'dog']) == {3: 2, 5: 1}


def test_countby_parity():
    assert countby(lambda x: x % 2 == 0, [1, 2, 3]) == {True: 1, False: 2}


def test_partitionby_chars():
    is_space = lambda c: c == " "
    assert list(partitionby(is_space, "I have space")) == [
        ('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')]


def test_partitionby_numbers():
    is_large = lambda x: x > 10
    assert list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5])) == [
        (1, 2, 1), (99, 88, 33, 99), (-1, 5)]


def test_unzip():
    a, b = unzip([('a', 1), ('b', 2)])
    assert list(a) == ['a', 'b']
    assert list(b) == [1, 2]


def test_equality_hash_key_unhashable_items():
    EqualityHashDefault = curry(EqualityHashKey, None)
    keys = set(map(EqualityHashDefault, [[], (), [1], [1]]))
    assert len(keys) == 3


def test_fold_sum_with_default():
    assert fold(add, list(range(10)), 0) == 45
