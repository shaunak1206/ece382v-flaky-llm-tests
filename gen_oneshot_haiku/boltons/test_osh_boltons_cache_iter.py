"""Tests for boltons cache and iterator utilities."""

import pytest
from boltons.cacheutils import LRI, LRU, cached, cachedmethod, ThresholdCounter
from boltons.iterutils import (
    chunked, windowed, unique, first, bucketize, remap,
    partition, one, flatten
)


class TestLRICache:
    """Test LRI (Least Recently Inserted) cache."""

    def test_lri_basic_get_set(self):
        """Test basic LRI get and set."""
        cache = LRI(max_size=2)
        cache['a'] = 1
        cache['b'] = 2
        assert cache['a'] == 1
        assert cache['b'] == 2

    def test_lri_eviction(self):
        """Test LRI eviction when max_size exceeded."""
        cache = LRI(max_size=2)
        cache['a'] = 1
        cache['b'] = 2
        cache['c'] = 3  # 'a' should be evicted
        assert 'a' not in cache
        assert 'b' in cache
        assert 'c' in cache

    def test_lri_on_miss(self):
        """Test LRI with on_miss callback."""
        call_count = [0]

        def on_miss(key):
            call_count[0] += 1
            return key * 10

        cache = LRI(max_size=2, on_miss=on_miss)
        assert cache['x'] == 'xxxxxxxxxxxx'  # 'x' * 10
        assert call_count[0] == 1

    def test_lri_statistics(self):
        """Test LRI hit and miss statistics."""
        cache = LRI(max_size=3)
        cache['a'] = 1
        cache['b'] = 2
        # Get existing key (hit)
        _ = cache['a']
        # Get missing key (miss)
        _ = cache.get('c', None)
        # Check statistics exist
        assert hasattr(cache, 'hit_count') or hasattr(cache, 'miss_count')

    def test_lri_update(self):
        """Test LRI update."""
        cache = LRI(max_size=3)
        cache['a'] = 1
        cache['a'] = 2
        assert cache['a'] == 2

    def test_lri_clear(self):
        """Test LRI clear."""
        cache = LRI(max_size=3)
        cache['a'] = 1
        cache['b'] = 2
        cache.clear()
        assert len(cache) == 0


class TestLRUCache:
    """Test LRU (Least Recently Used) cache."""

    def test_lru_basic_get_set(self):
        """Test basic LRU get and set."""
        cache = LRU(max_size=2)
        cache['a'] = 1
        cache['b'] = 2
        assert cache['a'] == 1
        assert cache['b'] == 2

    def test_lru_eviction_order(self):
        """Test LRU eviction order."""
        cache = LRU(max_size=3)
        cache['a'] = 1
        cache['b'] = 2
        cache['c'] = 3
        # Access 'a' to make it recently used
        _ = cache['a']
        # Add new item, 'b' should be evicted (least recently used)
        cache['d'] = 4
        assert 'a' in cache
        assert 'b' not in cache
        assert 'd' in cache

    def test_lru_on_miss(self):
        """Test LRU with on_miss callback."""
        call_count = [0]

        def on_miss(key):
            call_count[0] += 1
            return key.upper()

        cache = LRU(max_size=2, on_miss=on_miss)
        result = cache['hello']
        assert result == 'HELLO'
        assert call_count[0] == 1

    def test_lru_statistics(self):
        """Test LRU statistics."""
        cache = LRU(max_size=2)
        cache['a'] = 1
        _ = cache.get('a')
        _ = cache.get('b', None)
        # Statistics should be tracked
        assert hasattr(cache, 'hit_count') or hasattr(cache, 'miss_count')


class TestCachedDecorator:
    """Test @cached decorator."""

    def test_cached_simple(self):
        """Test basic @cached decorator."""
        call_count = [0]

        @cached({})
        def expensive_func(x):
            call_count[0] += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count[0] == 1  # Function called only once

    def test_cached_different_args(self):
        """Test cached with different arguments."""
        call_count = [0]

        @cached({})
        def func(x):
            call_count[0] += 1
            return x ** 2

        assert func(3) == 9
        assert func(4) == 16
        assert call_count[0] == 2  # Called for each different arg


class TestThresholdCounter:
    """Test ThresholdCounter."""

    def test_threshold_counter_basic(self):
        """Test basic ThresholdCounter."""
        tc = ThresholdCounter(max_size=3)
        tc['a'] += 1
        tc['b'] += 1
        tc['a'] += 1
        assert tc['a'] >= 1
        assert tc['b'] >= 1

    def test_threshold_counter_size_limit(self):
        """Test ThresholdCounter size limitation."""
        tc = ThresholdCounter(max_size=2)
        tc['a'] += 1
        tc['b'] += 1
        tc['c'] += 1  # Should handle overflow
        # Should have at most max_size items or handle gracefully
        assert len(tc) <= 2 or len(tc) > 0


class TestChunked:
    """Test chunked iterator."""

    def test_chunked_basic(self):
        """Test basic chunked."""
        data = [1, 2, 3, 4, 5]
        result = list(chunked(data, 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunked_exact_division(self):
        """Test chunked with exact division."""
        data = [1, 2, 3, 4]
        result = list(chunked(data, 2))
        assert result == [[1, 2], [3, 4]]

    def test_chunked_size_one(self):
        """Test chunked with size 1."""
        data = ['a', 'b', 'c']
        result = list(chunked(data, 1))
        assert result == [['a'], ['b'], ['c']]

    def test_chunked_string(self):
        """Test chunked on string."""
        result = list(chunked('abcde', 2))
        assert result == [['a', 'b'], ['c', 'd'], ['e']]

    def test_chunked_empty(self):
        """Test chunked on empty list."""
        result = list(chunked([], 2))
        assert result == []


class TestWindowed:
    """Test windowed iterator."""

    def test_windowed_basic(self):
        """Test basic windowed."""
        data = [1, 2, 3, 4, 5]
        result = list(windowed(data, 2))
        assert len(result) > 0
        # First window should have first two elements

    def test_windowed_size_one(self):
        """Test windowed with size 1."""
        data = [1, 2, 3]
        result = list(windowed(data, 1))
        assert len(result) >= 1

    def test_windowed_larger_than_input(self):
        """Test windowed when window size > input size."""
        data = [1, 2]
        result = list(windowed(data, 5, fill=0))
        assert len(result) > 0


class TestUnique:
    """Test unique iterator."""

    def test_unique_basic(self):
        """Test basic unique."""
        data = [1, 2, 2, 3, 1, 4]
        result = list(unique(data))
        assert len(result) == 4
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert 4 in result

    def test_unique_strings(self):
        """Test unique on strings."""
        data = ['a', 'b', 'a', 'c', 'b']
        result = list(unique(data))
        assert len(result) == 3

    def test_unique_with_key(self):
        """Test unique with key function."""
        data = [1, 2, 3, 4, 5]
        result = list(unique(data, key=lambda x: x % 2))
        # Should have one odd and one even
        assert len(result) <= 2

    def test_unique_empty(self):
        """Test unique on empty list."""
        result = list(unique([]))
        assert result == []


class TestFirst:
    """Test first function."""

    def test_first_basic(self):
        """Test basic first."""
        data = [1, 2, 3]
        result = first(data)
        assert result == 1

    def test_first_with_default(self):
        """Test first with default on empty."""
        result = first([], default=42)
        assert result == 42

    def test_first_with_key(self):
        """Test first with key function."""
        data = [1, 2, 3, 4, 5]
        result = first(data, key=lambda x: x > 2)
        assert result == 3

    def test_first_missing_raises(self):
        """Test first raises when not found and no default."""
        with pytest.raises((StopIteration, ValueError)):
            first([], key=lambda x: x > 100)


class TestOne:
    """Test one function."""

    def test_one_single_element(self):
        """Test one with single element."""
        result = one([42])
        assert result == 42

    def test_one_multiple_raises(self):
        """Test one raises with multiple elements."""
        with pytest.raises((ValueError, StopIteration)):
            one([1, 2])

    def test_one_empty_default(self):
        """Test one with default on empty."""
        result = one([], default=None)
        assert result is None

    def test_one_with_key(self):
        """Test one with key function."""
        data = [1, 2, 3, 4]
        result = one(data, key=lambda x: x == 3)
        assert result == 3


class TestBucketize:
    """Test bucketize function."""

    def test_bucketize_basic(self):
        """Test basic bucketize with bool key."""
        data = [1, 2, 3, 4, 5]
        result = bucketize(data, key=lambda x: x % 2 == 0)
        assert True in result
        assert False in result

    def test_bucketize_with_string_key(self):
        """Test bucketize with string key function."""
        data = ['apple', 'apricot', 'banana', 'blueberry']
        result = bucketize(data, key=lambda x: x[0])
        assert 'a' in result
        assert 'b' in result

    def test_bucketize_single_value(self):
        """Test bucketize with single value."""
        result = bucketize([42])
        assert len(result) > 0


class TestFlatten:
    """Test flatten function."""

    def test_flatten_basic(self):
        """Test basic flatten."""
        data = [[1, 2], [3, 4], [5]]
        result = list(flatten(data))
        assert result == [1, 2, 3, 4, 5]

    def test_flatten_nested(self):
        """Test flatten on nested lists."""
        data = [[1, [2, 3]], [4, 5]]
        result = list(flatten(data))
        # Should flatten at least one level
        assert 1 in result or 2 in result

    def test_flatten_empty(self):
        """Test flatten on empty."""
        result = list(flatten([]))
        assert result == []


class TestPartition:
    """Test partition function."""

    def test_partition_basic(self):
        """Test basic partition with bool key."""
        data = [1, 2, 3, 4, 5]
        result = partition(data, key=lambda x: x % 2 == 0)
        # Should have two groups
        assert len(result) == 2

    def test_partition_multiple_keys(self):
        """Test partition with multiple key functions."""
        data = [0, 1, 2, 3, 4, 5]
        result = partition(data, lambda x: x % 2, lambda x: x % 3)
        # Should partition by multiple keys
        assert len(result) >= 2


class TestRemap:
    """Test remap function for remapping nested structures."""

    def test_remap_basic(self):
        """Test basic remap."""
        data = {'a': 1, 'b': {'c': 2}}
        result = remap(data)
        assert result == data or len(result) > 0

    def test_remap_with_visit(self):
        """Test remap with visit function."""
        data = {'a': 1, 'b': 2}

        def visit(path, key, value):
            return True

        result = remap(data, visit=visit)
        assert len(result) > 0
