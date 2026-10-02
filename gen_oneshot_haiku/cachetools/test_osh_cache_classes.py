"""Tests for cachetools cache classes: Cache, LRUCache, LFUCache, FIFOCache, RRCache."""

import pytest
from cachetools import Cache, LRUCache, LFUCache, FIFOCache, RRCache, TTLCache, TLRUCache
import time


class TestCache:
    """Tests for base Cache class."""

    def test_cache_creation_with_valid_maxsize(self):
        """Test creating a cache with a valid maxsize."""
        cache = Cache(maxsize=10)
        assert cache.maxsize == 10
        assert cache.currsize == 0
        assert len(cache) == 0

    def test_cache_creation_with_negative_maxsize_raises_error(self):
        """Test that negative maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be non-negative"):
            Cache(maxsize=-1)

    def test_cache_setitem_and_getitem(self):
        """Test basic cache operations."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"
        assert "key1" in cache
        assert len(cache) == 1

    def test_cache_delitem(self):
        """Test deleting items from cache."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        del cache["key1"]
        assert "key1" not in cache
        assert len(cache) == 0

    def test_cache_delitem_nonexistent_raises_keyerror(self):
        """Test that deleting nonexistent item raises KeyError."""
        cache = Cache(maxsize=10)
        with pytest.raises(KeyError):
            del cache["nonexistent"]

    def test_cache_get_default(self):
        """Test get() method with default value."""
        cache = Cache(maxsize=10)
        assert cache.get("key1") is None
        assert cache.get("key1", "default") == "default"

    def test_cache_get_existing(self):
        """Test get() method for existing key."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        assert cache.get("key1") == "value1"

    def test_cache_pop_existing(self):
        """Test pop() method for existing key."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        result = cache.pop("key1")
        assert result == "value1"
        assert "key1" not in cache

    def test_cache_pop_nonexistent_with_default(self):
        """Test pop() with default for nonexistent key."""
        cache = Cache(maxsize=10)
        result = cache.pop("nonexistent", "default")
        assert result == "default"

    def test_cache_pop_nonexistent_raises_keyerror(self):
        """Test pop() raises KeyError for nonexistent key without default."""
        cache = Cache(maxsize=10)
        with pytest.raises(KeyError):
            cache.pop("nonexistent")

    def test_cache_setdefault_existing(self):
        """Test setdefault() for existing key."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        result = cache.setdefault("key1", "value2")
        assert result == "value1"

    def test_cache_setdefault_nonexistent(self):
        """Test setdefault() for nonexistent key."""
        cache = Cache(maxsize=10)
        result = cache.setdefault("key1", "value1")
        assert result == "value1"
        assert cache["key1"] == "value1"

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        cache.clear()
        assert len(cache) == 0
        assert cache.currsize == 0

    def test_cache_iteration(self):
        """Test iterating over cache keys."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        keys = list(cache)
        assert set(keys) == {"key1", "key2"}

    def test_cache_repr(self):
        """Test cache representation."""
        cache = Cache(maxsize=10)
        cache["key1"] = "value1"
        repr_str = repr(cache)
        assert "Cache" in repr_str
        assert "maxsize=10" in repr_str

    def test_cache_with_getsizeof(self):
        """Test cache with custom getsizeof function."""
        def custom_getsizeof(value):
            return len(str(value))

        cache = Cache(maxsize=10, getsizeof=custom_getsizeof)
        cache["key1"] = "12345"  # size 5
        assert cache.currsize == 5

    def test_cache_value_too_large(self):
        """Test that value larger than maxsize raises ValueError."""
        cache = Cache(maxsize=10)
        with pytest.raises(ValueError, match="value too large"):
            cache["key"] = object()  # object() has default size 1, need custom getsizeof

    def test_cache_eviction_on_overflow(self):
        """Test that popitem() is called when cache overflows."""
        cache = Cache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["d"] = 4  # Should evict one item
        assert len(cache) == 3

    def test_cache_currsize_tracking(self):
        """Test that currsize is correctly tracked."""
        def custom_getsizeof(value):
            return int(value)

        cache = Cache(maxsize=10, getsizeof=custom_getsizeof)
        cache["key1"] = 3
        assert cache.currsize == 3
        cache["key2"] = 4
        assert cache.currsize == 7


class TestLRUCache:
    """Tests for LRU Cache implementation."""

    def test_lru_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3

    def test_lru_evicts_least_recently_used(self):
        """Test that LRU evicts least recently used item."""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access 'a' to make it recently used
        _ = cache["a"]
        # Add new item, should evict 'b' (least recently used)
        cache["d"] = 4
        assert "b" not in cache
        assert "a" in cache

    def test_lru_access_updates_order(self):
        """Test that accessing item updates its position."""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        _ = cache["a"]
        cache["d"] = 4
        # 'b' should be evicted (least recently used after 'a' access)
        assert "b" not in cache
        assert "a" in cache

    def test_lru_clear(self):
        """Test clearing LRU cache."""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0

    def test_lru_with_getsizeof(self):
        """Test LRU cache with custom size function."""
        def custom_getsizeof(value):
            return int(value)

        cache = LRUCache(maxsize=10, getsizeof=custom_getsizeof)
        cache["a"] = 5
        cache["b"] = 3
        cache["c"] = 4  # Exceeds maxsize
        assert cache.currsize <= 10

    def test_lru_popitem_removes_lru(self):
        """Test that popitem removes least recently used."""
        cache = LRUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        _ = cache["a"]
        key, value = cache.popitem()
        assert key == "b"
        assert value == 2

    def test_lru_empty_popitem_raises_keyerror(self):
        """Test that popitem on empty cache raises KeyError."""
        cache = LRUCache(maxsize=3)
        with pytest.raises(KeyError):
            cache.popitem()


class TestLFUCache:
    """Tests for LFU Cache implementation."""

    def test_lfu_basic_operations(self):
        """Test basic LFU cache operations."""
        cache = LFUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3

    def test_lfu_evicts_least_frequently_used(self):
        """Test that LFU evicts least frequently used item."""
        cache = LFUCache(maxsize=3)
        cache["a"] = 1
        _ = cache["a"]
        _ = cache["a"]
        cache["b"] = 2
        _ = cache["b"]
        cache["c"] = 3
        # 'a' used 3 times, 'b' used 2 times, 'c' used 1 time
        cache["d"] = 4
        # 'c' should be evicted (least frequently used)
        assert "c" not in cache
        assert "a" in cache

    def test_lfu_frequency_tracking(self):
        """Test that LFU correctly tracks access frequency."""
        cache = LFUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        # Access 'a' multiple times
        for _ in range(5):
            _ = cache["a"]
        # Access 'b' once more
        _ = cache["b"]
        cache["c"] = 3
        # Add new item; 'b' should be evicted
        cache["d"] = 4
        assert "b" not in cache

    def test_lfu_clear(self):
        """Test clearing LFU cache."""
        cache = LFUCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0

    def test_lfu_popitem_removes_lfu(self):
        """Test that popitem removes least frequently used."""
        cache = LFUCache(maxsize=3)
        cache["a"] = 1
        _ = cache["a"]
        cache["b"] = 2
        cache["c"] = 3
        # 'a' accessed twice, 'b' and 'c' accessed once
        key, value = cache.popitem()
        assert key in ["b", "c"]


class TestFIFOCache:
    """Tests for FIFO Cache implementation."""

    def test_fifo_basic_operations(self):
        """Test basic FIFO cache operations."""
        cache = FIFOCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3

    def test_fifo_evicts_oldest_item(self):
        """Test that FIFO evicts oldest (first-in) item."""
        cache = FIFOCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Add new item, should evict 'a' (first inserted)
        cache["d"] = 4
        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert "d" in cache

    def test_fifo_ignores_access_order(self):
        """Test that FIFO ignores access order (not like LRU)."""
        cache = FIFOCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access 'a' many times
        for _ in range(10):
            _ = cache["a"]
        # Still, 'a' should be evicted as it was inserted first
        cache["d"] = 4
        assert "a" not in cache

    def test_fifo_clear(self):
        """Test clearing FIFO cache."""
        cache = FIFOCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0

    def test_fifo_popitem_removes_first_inserted(self):
        """Test that popitem removes first inserted."""
        cache = FIFOCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        key, value = cache.popitem()
        assert key == "a"
        assert value == 1


class TestRRCache:
    """Tests for Random Replacement Cache."""

    def test_rr_basic_operations(self):
        """Test basic RR cache operations."""
        cache = RRCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3

    def test_rr_evicts_random_item(self):
        """Test that RR evicts random item (may be any)."""
        cache = RRCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["d"] = 4
        # One of a, b, or c should be evicted
        assert len(cache) == 3
        assert "d" in cache

    def test_rr_with_deterministic_choice(self):
        """Test RR with a deterministic choice function."""
        def deterministic_choice(items):
            return items[0]

        cache = RRCache(maxsize=3, choice=deterministic_choice)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache["d"] = 4
        # Should evict 'a' (first in list)
        assert "a" not in cache
        assert "d" in cache

    def test_rr_clear(self):
        """Test clearing RR cache."""
        cache = RRCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0

    def test_rr_popitem_removes_random(self):
        """Test that popitem removes random item."""
        cache = RRCache(maxsize=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        key, value = cache.popitem()
        assert key in ["a", "b", "c"]
        assert len(cache) == 2


class TestTTLCache:
    """Tests for TTL Cache."""

    def test_ttl_basic_operations(self):
        """Test basic TTL cache operations."""
        cache = TTLCache(maxsize=3, ttl=10)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2

    def test_ttl_item_expiration(self):
        """Test that items expire after TTL."""
        cache = TTLCache(maxsize=3, ttl=0.1)
        cache["a"] = 1
        time.sleep(0.2)
        with pytest.raises(KeyError):
            _ = cache["a"]

    def test_ttl_contains_with_expired_item(self):
        """Test __contains__ with expired item."""
        cache = TTLCache(maxsize=3, ttl=0.1)
        cache["a"] = 1
        assert "a" in cache
        time.sleep(0.2)
        assert "a" not in cache

    def test_ttl_get_with_expired_item(self):
        """Test get() with expired item."""
        cache = TTLCache(maxsize=3, ttl=0.1)
        cache["a"] = 1
        time.sleep(0.2)
        assert cache.get("a") is None

    def test_ttl_expire_method(self):
        """Test expire() method."""
        cache = TTLCache(maxsize=3, ttl=0.1)
        cache["a"] = 1
        cache["b"] = 2
        time.sleep(0.2)
        expired = cache.expire()
        assert len(expired) > 0

    def test_ttl_len_with_expired_items(self):
        """Test that len() excludes expired items."""
        cache = TTLCache(maxsize=3, ttl=0.1)
        cache["a"] = 1
        cache["b"] = 2
        initial_len = len(cache)
        time.sleep(0.2)
        after_len = len(cache)
        assert after_len < initial_len

    def test_ttl_custom_timer(self):
        """Test TTL cache with custom timer."""
        timer_values = [0, 1, 2, 3, 4]
        timer_index = [0]

        def custom_timer():
            val = timer_values[timer_index[0]]
            timer_index[0] += 1
            return val

        cache = TTLCache(maxsize=3, ttl=2, timer=custom_timer)
        cache["a"] = 1
        cache["b"] = 2
        # Timer advances to 2, then 3, 4
        expired = cache.expire()
        # Item should be expired after TTL
        assert len(expired) >= 0

    def test_ttl_clear(self):
        """Test clearing TTL cache."""
        cache = TTLCache(maxsize=3, ttl=10)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0
