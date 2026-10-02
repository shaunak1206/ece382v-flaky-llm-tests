"""Tests for the basic Cache class and the LRU/LFU/FIFO/RR eviction policies."""

import random

import pytest

from cachetools import Cache, FIFOCache, LFUCache, LRUCache, RRCache

ALL_CACHE_CLASSES = [Cache, FIFOCache, LFUCache, LRUCache, RRCache]


# ---------------------------------------------------------------------------
# Cache (base class)
# ---------------------------------------------------------------------------


def test_cache_basic_mapping_operations():
    with pytest.raises(ValueError):
        Cache(maxsize=-1)

    cache = Cache(maxsize=5)
    cache["a"] = 1
    cache[2] = "two"
    assert cache["a"] == 1
    assert len(cache) == 2
    assert set(cache) == {"a", 2}
    assert "a" in cache
    assert "zzz" not in cache
    del cache["a"]
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_cache_get_pop_setdefault():
    cache = Cache(maxsize=4)
    cache["x"] = 10
    assert cache.get("x") == 10
    assert cache.get("y") is None
    assert cache.get("y", "fallback") == "fallback"
    assert cache.pop("x") == 10
    assert cache.pop("x", "gone") == "gone"
    with pytest.raises(KeyError):
        cache.pop("x")
    assert cache.setdefault("k", 1) == 1
    assert cache.setdefault("k", 2) == 1
    assert cache["k"] == 1


def test_cache_getsizeof_tracks_currsize():
    cache = Cache(maxsize=10, getsizeof=len)
    cache["a"] = "abc"
    assert cache.currsize == 3
    cache["b"] = "de"
    assert cache.currsize == 5
    cache["a"] = "abcdef"  # overwrite grows the entry
    assert cache.currsize == 8
    cache["a"] = "x"
    assert cache.currsize == 3
    del cache["b"]
    assert cache.currsize == 1
    assert cache.maxsize == 10


def test_cache_rejects_oversized_and_invalid_sizes():
    small = Cache(maxsize=2, getsizeof=len)
    with pytest.raises(ValueError):
        small["a"] = "abc"
    assert "a" not in small
    assert small.currsize == 0

    negative = Cache(maxsize=5, getsizeof=lambda value: -1)
    with pytest.raises(ValueError):
        negative["a"] = "whatever"

    zero = Cache(maxsize=0)
    with pytest.raises(ValueError):
        zero[1] = 1
    assert len(zero) == 0


def test_cache_clear_and_repr():
    cache = Cache(maxsize=3)
    assert repr(cache) == "Cache({}, maxsize=3, currsize=0)"
    cache[1] = 2
    assert repr(cache) == "Cache({1: 2}, maxsize=3, currsize=1)"
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    assert list(cache) == []


def test_cache_subclass_missing_hook():
    class Doubling(Cache):
        def __missing__(self, key):
            value = self[key] = key * 2
            return value

    cache = Doubling(maxsize=5)
    assert cache[3] == 6
    assert 3 in cache
    # get() must not trigger __missing__
    assert cache.get(7) is None
    assert 7 not in cache


@pytest.mark.parametrize("cls", ALL_CACHE_CLASSES)
def test_all_policies_never_exceed_maxsize(cls):
    cache = cls(3)
    for i in range(10):
        cache[i] = i
        assert len(cache) <= 3
    assert len(cache) == 3
    assert cache.currsize == 3
    assert 9 in cache


@pytest.mark.parametrize("cls", ALL_CACHE_CLASSES)
def test_all_policies_update_clear_and_delete(cls):
    cache = cls(2)
    cache[1] = "a"
    cache[2] = "b"
    cache[1] = "c"  # updating must not evict anything
    assert len(cache) == 2
    assert cache[1] == "c"
    assert cache[2] == "b"
    with pytest.raises(KeyError):
        del cache["missing"]
    cache.clear()
    assert len(cache) == 0
    cache[3] = 3
    cache[4] = 4
    assert set(cache) == {3, 4}


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_evicts_least_recently_used():
    cache = LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # touch a, so b becomes the LRU entry
    cache["c"] = 3
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache

    # overwriting also refreshes recency
    cache["a"] = 10
    cache["d"] = 4
    assert "c" not in cache
    assert cache["a"] == 10


def test_lru_popitem_order_and_empty():
    cache = LRUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"]
    assert cache.popitem() == ("b", 2)
    assert cache.popitem() == ("c", 3)
    assert cache.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lru_getsizeof_evicts_multiple_entries():
    cache = LRUCache(maxsize=5, getsizeof=len)
    cache["a"] = "aa"
    cache["b"] = "bb"
    cache["c"] = "ccc"
    assert "a" not in cache
    assert cache.currsize == 5
    cache["d"] = "dddd"
    assert list(cache) == ["d"]
    assert cache.currsize == 4


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------


def test_lfu_evicts_least_frequently_used():
    cache = LFUCache(maxsize=2)
    cache[1] = "one"
    cache[2] = "two"
    cache[1]
    cache[1]
    cache[3] = "three"
    assert 2 not in cache
    assert 1 in cache
    assert 3 in cache


def test_lfu_popitem_order_and_empty():
    cache = LFUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["a"]
    cache["a"]
    cache["b"]
    assert cache.popitem() == ("c", 3)
    assert cache.popitem() == ("b", 2)
    assert cache.popitem() == ("a", 1)
    with pytest.raises(KeyError):
        cache.popitem()


def test_lfu_delete_and_reinsert_resets_frequency():
    cache = LFUCache(maxsize=2)
    cache["a"] = 1
    cache["a"]
    cache["a"]
    cache["b"] = 2
    del cache["a"]
    cache["a"] = 3
    cache["b"]  # b is now used more often than the fresh a
    cache["c"] = 4
    assert set(cache) == {"b", "c"}


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------


def test_fifo_evicts_oldest_regardless_of_access():
    cache = FIFOCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]
    cache["a"]
    cache["c"] = 3
    assert "a" not in cache
    assert set(cache) == {"b", "c"}


def test_fifo_popitem_order_and_update_moves_key_to_end():
    cache = FIFOCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache.popitem() == ("a", 1)
    assert cache.popitem() == ("b", 2)
    assert cache.popitem() == ("c", 3)
    with pytest.raises(KeyError):
        cache.popitem()

    cache = FIFOCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 10
    cache["c"] = 3
    assert "b" not in cache
    assert cache["a"] == 10


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------


def test_rr_custom_choice_is_deterministic():
    def first(keys):
        return keys[0]

    cache = RRCache(2, choice=first)
    assert cache.choice is first
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert set(cache) == {"b", "c"}
    assert RRCache(2).choice is random.choice


def test_rr_eviction_is_random():
    evicted = set()
    for _ in range(100):
        cache = RRCache(3)
        for key in "abc":
            cache[key] = ord(key)
        cache["d"] = 0
        missing = {"a", "b", "c"} - set(cache)
        assert len(missing) == 1
        evicted |= missing
    # With 100 trials and 3 candidates, more than one key should get evicted.
    assert len(evicted) > 1


def test_rr_never_exceeds_maxsize_and_is_reproducible_when_seeded():
    cache = RRCache(5)
    for i in range(100):
        cache[i] = i
        assert len(cache) <= 5
    assert cache.currsize == 5

    results = []
    for _ in range(2):
        random.seed(1234)
        seeded = RRCache(3)
        for i in range(10):
            seeded[i] = i
        results.append(sorted(seeded))
    assert results[0] == results[1]


def test_rr_delete_keeps_index_consistent_and_popitem_empty():
    cache = RRCache(4)
    for i in range(4):
        cache[i] = i * 10
    del cache[1]
    del cache[0]
    assert len(cache) == 2
    key, value = cache.popitem()
    assert value == key * 10
    assert key not in cache
    cache.popitem()
    with pytest.raises(KeyError):
        cache.popitem()
