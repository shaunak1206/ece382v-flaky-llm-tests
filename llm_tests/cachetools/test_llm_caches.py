"""Tests for the non-time-based cache classes: Cache, LRUCache, LFUCache,
FIFOCache, RRCache.
"""

import pytest

from cachetools import Cache, FIFOCache, LFUCache, LRUCache, RRCache

CACHE_CLASSES = [Cache, FIFOCache, LFUCache, LRUCache, RRCache]


@pytest.fixture(params=CACHE_CLASSES)
def cache_class(request):
    return request.param


def test_negative_maxsize_raises(cache_class):
    with pytest.raises(ValueError):
        cache_class(-1)


def test_basic_set_get(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    assert cache["a"] == 1


def test_missing_key_raises_keyerror(cache_class):
    cache = cache_class(maxsize=2)
    with pytest.raises(KeyError):
        cache["missing"]


def test_contains(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    assert "a" in cache
    assert "b" not in cache


def test_len_and_currsize(cache_class):
    cache = cache_class(maxsize=10)
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["a"] = 1
    cache["b"] = 2
    assert len(cache) == 2
    assert cache.currsize == 2


def test_delitem(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache
    with pytest.raises(KeyError):
        del cache["a"]


def test_pop(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    assert cache.pop("a") == 1
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache.pop("a")
    assert cache.pop("a", "default") == "default"


def test_setdefault(cache_class):
    cache = cache_class(maxsize=2)
    assert cache.setdefault("a", 1) == 1
    assert cache["a"] == 1
    # existing key: does not overwrite
    assert cache.setdefault("a", 2) == 1
    assert cache["a"] == 1


def test_clear(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    assert "a" not in cache


def test_value_larger_than_maxsize_raises(cache_class):
    cache = cache_class(maxsize=2, getsizeof=lambda v: v)
    with pytest.raises(ValueError):
        cache["a"] = 3


def test_getsizeof_affects_currsize(cache_class):
    cache = cache_class(maxsize=10, getsizeof=lambda v: len(v))
    cache["a"] = "xx"
    cache["b"] = "xxxx"
    assert cache.currsize == 6


def test_eviction_happens_when_full(cache_class):
    cache = cache_class(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert len(cache) == 2
    assert cache.currsize == 2


def test_popitem_on_empty_raises(cache_class):
    cache = cache_class(maxsize=2)
    with pytest.raises(KeyError):
        cache.popitem()


# --- Cache base class specifics ---


def test_cache_base_default_getsizeof_returns_one():
    assert Cache.getsizeof("anything") == 1


def test_cache_base_popitem_uses_mutable_mapping_default():
    # Cache itself does not override popitem(); it falls back to the
    # collections.abc.MutableMapping mixin implementation.
    cache = Cache(maxsize=2)
    cache["a"] = 1
    key, value = cache.popitem()
    assert (key, value) == ("a", 1)
    assert "a" not in cache


# --- FIFOCache specifics ---


def test_fifo_evicts_first_inserted():
    cache = FIFOCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3  # should evict "a"
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


# --- LRUCache specifics ---


def test_lru_evicts_least_recently_used():
    cache = LRUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # touch "a", making "b" the LRU
    cache["c"] = 3  # should evict "b"
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


# --- LFUCache specifics ---


def test_lfu_evicts_least_frequently_used():
    cache = LFUCache(maxsize=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]
    cache["a"]  # "a" accessed more often now
    cache["c"] = 3  # should evict "b" (least frequently used)
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


# --- RRCache specifics ---


def test_rr_choice_function_is_used():
    calls = []

    def choice(seq):
        calls.append(list(seq))
        return seq[0]

    cache = RRCache(maxsize=2, choice=choice)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3  # triggers eviction via choice()
    assert len(calls) == 1
    assert len(cache) == 2


def test_rr_evicts_a_key_deterministically_with_fixed_choice():
    # always evict the first candidate in internal key order
    cache = RRCache(maxsize=2, choice=lambda seq: seq[0])
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert len(cache) == 2
    assert "c" in cache  # newest item always present
