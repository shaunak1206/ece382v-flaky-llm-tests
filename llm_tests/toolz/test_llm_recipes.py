"""LLM-generated unit tests for toolz.recipes and toolz.sandbox, based on
source/docstrings only."""
from toolz import curry
from toolz.recipes import countby, partitionby
from toolz.sandbox.core import EqualityHashKey, unzip


def test_countby_by_len():
    assert countby(len, ['cat', 'mouse', 'dog']) == {3: 2, 5: 1}


def test_countby_by_predicate():
    def iseven(x):
        return x % 2 == 0
    result = countby(iseven, [1, 2, 3])
    assert result == {True: 1, False: 2}


def test_partitionby_on_spaces():
    is_space = lambda c: c == " "
    result = list(partitionby(is_space, "I have space"))
    assert result == [
        ('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',),
        ('s', 'p', 'a', 'c', 'e'),
    ]


def test_partitionby_on_threshold():
    is_large = lambda x: x > 10
    result = list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5]))
    assert result == [(1, 2, 1), (99, 88, 33, 99), (-1, 5)]


def test_unzip_pairs():
    a, b = unzip([('a', 1), ('b', 2)])
    assert list(a) == ['a', 'b']
    assert list(b) == [1, 2]


def test_unzip_empty():
    result = unzip([])
    assert result == tuple()


def test_equality_hash_key_dedupes_unhashable_lists():
    from toolz import unique
    equality_hash_default = curry(EqualityHashKey, None)
    vals = [[], [], (), [1], [1], [2], {}, {}, {}]
    result = list(unique(vals, key=equality_hash_default))
    assert result == [[], (), [1], [2], {}]


def test_equality_hash_key_with_len_key():
    equality_hash_len = curry(EqualityHashKey, len)
    vals = [[], (), [1], [1]]
    keys = set(map(equality_hash_len, vals))
    assert len(keys) == 3  # [] and () both have len 0, so they collide


def test_equality_hash_key_equality_and_repr():
    k1 = EqualityHashKey(None, [1, 2])
    k2 = EqualityHashKey(None, [1, 2])
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert repr(k1) == '=[1, 2]='
