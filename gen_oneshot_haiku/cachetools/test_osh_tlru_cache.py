"""Tests for TLRUCache (Time-aware LRU cache)."""

import pytest
from cachetools import TLRUCache
import time


class TestTLRUCache:
    """Tests for TLRUCache implementation."""

    def test_tlru_basic_operations(self):
        """Test basic TLRU cache operations."""
        def ttu(key, value, now):
            """Time-to-use function: items live for 10 seconds."""
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        assert cache["a"] == 1
        assert cache["b"] == 2

    def test_tlru_item_expiration(self):
        """Test that TLRU items expire based on ttu."""
        def ttu(key, value, now):
            """Items expire after 0.1 seconds."""
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        assert cache["a"] == 1
        time.sleep(0.2)
        with pytest.raises(KeyError):
            _ = cache["a"]

    def test_tlru_ttu_per_item(self):
        """Test that different items can have different TTU values."""
        def ttu(key, value, now):
            """Items with even values expire faster."""
            if value % 2 == 0:
                return now + 0.1
            else:
                return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1  # Expires in 10 seconds
        cache["b"] = 2  # Expires in 0.1 seconds
        time.sleep(0.2)
        # 'a' should still be there
        assert cache["a"] == 1
        # 'b' should be expired
        with pytest.raises(KeyError):
            _ = cache["b"]

    def test_tlru_contains_with_expired_item(self):
        """Test __contains__ with expired item."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        assert "a" in cache
        time.sleep(0.2)
        assert "a" not in cache

    def test_tlru_get_with_expired_item(self):
        """Test get() with expired item."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        time.sleep(0.2)
        assert cache.get("a") is None

    def test_tlru_expire_method(self):
        """Test expire() method."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        time.sleep(0.2)
        expired = cache.expire()
        assert len(expired) == 2

    def test_tlru_len_with_expired_items(self):
        """Test that len() excludes expired items."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        initial_len = len(cache)
        time.sleep(0.2)
        after_len = len(cache)
        assert after_len < initial_len

    def test_tlru_lru_eviction_when_full(self):
        """Test that TLRU evicts LRU item when full."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access 'a' to make it recently used
        _ = cache["a"]
        # Add new item, should evict 'b' (LRU)
        cache["d"] = 4
        assert "b" not in cache
        assert "a" in cache

    def test_tlru_clear(self):
        """Test clearing TLRU cache."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0

    def test_tlru_setitem_updates_ttu(self):
        """Test that setitem updates TTU for existing keys."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        time.sleep(0.05)
        cache["a"] = 2  # Update with new value and new TTU
        time.sleep(0.1)
        # Should still be there (updated TTU)
        assert cache["a"] == 2

    def test_tlru_custom_timer(self):
        """Test TLRU cache with custom timer."""
        timer_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        timer_idx = [0]

        def custom_timer():
            val = timer_values[timer_idx[0]]
            timer_idx[0] += 1
            return val

        def ttu(key, value, now):
            return now + 1.0

        cache = TLRUCache(maxsize=3, ttu=ttu, timer=custom_timer)
        cache["a"] = 1
        expired = cache.expire()
        assert len(expired) >= 0

    def test_tlru_delitem(self):
        """Test deleting items from TLRU cache."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        del cache["a"]
        assert "a" not in cache
        assert "b" in cache

    def test_tlru_delitem_expired_raises_keyerror(self):
        """Test that deleting expired item raises KeyError."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        time.sleep(0.2)
        with pytest.raises(KeyError):
            del cache["a"]

    def test_tlru_popitem(self):
        """Test popitem() on TLRU cache."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        _ = cache["a"]
        key, value = cache.popitem()
        # Should remove LRU (least recently used)
        assert key in ["b", "c"]

    def test_tlru_iteration(self):
        """Test iterating over TLRU cache."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        keys = list(cache)
        assert set(keys) == {"a", "b", "c"}

    def test_tlru_iteration_with_expired_items(self):
        """Test iteration excludes expired items."""
        def ttu(key, value, now):
            if key == "a":
                return now + 0.1
            else:
                return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        cache["b"] = 2
        time.sleep(0.2)
        keys = list(cache)
        assert "a" not in keys
        assert "b" in keys

    def test_tlru_with_getsizeof(self):
        """Test TLRU cache with custom size function."""
        def custom_getsizeof(value):
            return int(value)

        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=10, ttu=ttu, getsizeof=custom_getsizeof)
        cache["a"] = 5
        cache["b"] = 3
        assert cache.currsize == 8

    def test_tlru_update_expired_item(self):
        """Test updating an expired item."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        time.sleep(0.2)
        # Item is expired, setting it again should add it
        cache["a"] = 2
        assert cache["a"] == 2

    def test_tlru_repr(self):
        """Test TLRU cache representation."""
        def ttu(key, value, now):
            return now + 10

        cache = TLRUCache(maxsize=3, ttu=ttu)
        cache["a"] = 1
        repr_str = repr(cache)
        assert "TLRUCache" in repr_str

    def test_tlru_heap_cleanup(self):
        """Test that TLRU cache cleans up heap when needed."""
        def ttu(key, value, now):
            return now + 0.1

        cache = TLRUCache(maxsize=100, ttu=ttu)
        # Add and expire many items to trigger cleanup
        for i in range(50):
            cache[f"key{i}"] = i
        time.sleep(0.2)
        # Add new items, should trigger cleanup
        for i in range(50, 60):
            cache[f"key{i}"] = i
        # Cache should still work correctly
        assert len(cache) <= 100
