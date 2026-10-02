import pytest
from toolz.recipes import countby, partitionby
from toolz.sandbox.core import unzip, EqualityHashKey
from toolz import curry


class TestCountby:
    def test_countby_by_length(self):
        result = countby(len, ['cat', 'mouse', 'dog'])
        assert result == {3: 2, 5: 1}

    def test_countby_by_function(self):
        def iseven(x):
            return x % 2 == 0
        result = countby(iseven, [1, 2, 3])
        assert result == {True: 1, False: 2}

    def test_countby_empty(self):
        result = countby(len, [])
        assert result == {}

    def test_countby_single(self):
        result = countby(len, ['a'])
        assert result == {1: 1}

    def test_countby_by_key_string(self):
        data = [{'type': 'A'}, {'type': 'B'}, {'type': 'A'}]
        result = countby('type', data)
        assert result == {'A': 2, 'B': 1}


class TestPartitionby:
    def test_partitionby_space(self):
        is_space = lambda c: c == " "
        result = list(partitionby(is_space, "I have space"))
        assert result == [('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')]

    def test_partitionby_large(self):
        is_large = lambda x: x > 10
        result = list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5]))
        assert result == [(1, 2, 1), (99, 88, 33, 99), (-1, 5)]

    def test_partitionby_alternating(self):
        is_even = lambda x: x % 2 == 0
        result = list(partitionby(is_even, [1, 3, 2, 4, 5, 7]))
        assert result == [(1, 3), (2, 4), (5, 7)]

    def test_partitionby_single_group(self):
        is_even = lambda x: x % 2 == 0
        result = list(partitionby(is_even, [1, 3, 5, 7]))
        assert result == [(1, 3, 5, 7)]

    def test_partitionby_empty(self):
        result = list(partitionby(lambda x: x, []))
        assert result == []


class TestUnzip:
    def test_unzip_basic(self):
        a, b = unzip([('a', 1), ('b', 2)])
        assert list(a) == ['a', 'b']
        assert list(b) == [1, 2]

    def test_unzip_three_elements(self):
        a, b, c = unzip([(1, 2, 3), (4, 5, 6)])
        assert list(a) == [1, 4]
        assert list(b) == [2, 5]
        assert list(c) == [3, 6]

    def test_unzip_empty(self):
        result = unzip([])
        assert result == ()

    def test_unzip_single(self):
        a, b = unzip([('x', 'y')])
        assert list(a) == ['x']
        assert list(b) == ['y']


class TestEqualityHashKey:
    def test_equality_hash_key_basic(self):
        key1 = EqualityHashKey(None, [1, 2])
        key2 = EqualityHashKey(None, [1, 2])
        assert key1 == key2

    def test_equality_hash_key_inequality(self):
        key1 = EqualityHashKey(None, [1, 2])
        key2 = EqualityHashKey(None, [1, 3])
        assert key1 != key2

    def test_equality_hash_key_in_set(self):
        key1 = EqualityHashKey(None, [1, 2])
        key2 = EqualityHashKey(None, [1, 2])
        s = {key1}
        assert key2 in s

    def test_equality_hash_key_with_key_function(self):
        key1 = EqualityHashKey(len, [1, 2])
        key2 = EqualityHashKey(len, ['a', 'b'])
        # Different key values due to key function
        assert key1 != key2

    def test_equality_hash_key_str_repr(self):
        key = EqualityHashKey(None, [1, 2])
        assert str(key) == '=[1, 2]='
        assert repr(key) == '=[1, 2]='

    def test_equality_hash_key_unique_items(self):
        vals = [[], [], (), [1], [1], [2], {}, {}, {}]
        from toolz import unique
        EqualityHashDefault = curry(EqualityHashKey, None)
        result = list(unique(vals, key=EqualityHashDefault))
        # Result should have unique values based on equality
        assert len(result) == 4  # [], (), [1], [2], {}


class TestEqualityHashKeyWithKeyFunc:
    def test_equality_hash_key_with_len_key(self):
        EqualityHashLen = curry(EqualityHashKey, len)
        key1 = EqualityHashLen([1, 2])
        key2 = EqualityHashLen(['a', 'b'])
        # Both have length 2, so should hash the same
        assert hash(key1) == hash(key2)

    def test_equality_hash_key_with_type_key(self):
        EqualityHashType = curry(EqualityHashKey, type)
        key1 = EqualityHashType([])
        key2 = EqualityHashType(['a'])
        # Both are lists, so should hash the same
        assert hash(key1) == hash(key2)


class TestRecipesIntegration:
    def test_countby_partitionby_combo(self):
        # Test combining countby and partitionby concepts
        def iseven(x):
            return x % 2 == 0
        data = [1, 2, 3, 4, 5, 6]

        # Count by even/odd
        counts = countby(iseven, data)
        assert counts[True] == 3
        assert counts[False] == 3

        # Partition by even/odd
        partitions = list(partitionby(iseven, data))
        assert len(partitions) == 6  # Changes at each step


class TestSandboxIntegration:
    def test_unzip_with_unique(self):
        from toolz import unique
        data = [(1, 'a'), (1, 'a'), (2, 'b')]
        unique_data = list(unique(data))
        a, b = unzip(unique_data)
        assert len(list(a)) == 2

    def test_equality_hash_key_with_groupby(self):
        from toolz import groupby
        EqualityHashDefault = curry(EqualityHashKey, None)
        data = [[1, 2], [1, 2], [3, 4]]

        # Create hash keys for lists
        keyed = [EqualityHashDefault(item) for item in data]

        # Group by the key itself (identity)
        groups = groupby(lambda x: x, keyed)
        # Should have 2 groups (one for [1,2] repeated, one for [3,4])
        assert len(groups) == 2


class TestRecipesEdgeCases:
    def test_countby_with_none_values(self):
        result = countby(lambda x: x is None, [1, None, 2, None, 3])
        assert result == {True: 2, False: 3}

    def test_partitionby_single_element(self):
        result = list(partitionby(lambda x: x, [5]))
        assert result == [(5,)]

    def test_partitionby_all_same(self):
        result = list(partitionby(lambda x: x % 2 == 0, [2, 4, 6, 8]))
        assert result == [(2, 4, 6, 8)]

    def test_unzip_preserves_order(self):
        data = [(1, 'a'), (2, 'b'), (3, 'c')]
        a, b = unzip(data)
        assert list(a) == [1, 2, 3]
        assert list(b) == ['a', 'b', 'c']


class TestCurriedFunctions:
    def test_countby_curried(self):
        from toolz.curried import countby as countby_curried
        # Test that curried version works
        count_by_len = countby_curried(len)
        result = count_by_len(['a', 'bb', 'ccc', 'dd'])
        assert result == {1: 1, 2: 2, 3: 1}

    def test_partitionby_curried(self):
        from toolz.curried import partitionby as partitionby_curried
        # Test that curried version works
        is_even = lambda x: x % 2 == 0
        partition_by_even = partitionby_curried(is_even)
        result = list(partition_by_even([1, 2, 3, 4, 5]))
        assert len(result) > 1
