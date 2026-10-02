"""Tests for TTLCache and TLRUCache, using a manually-controlled timer so
that expiry behaviour is deterministic and fast (no real time.sleep()).
"""

import pytest

from cachetools import TLRUCache, TTLCache


class ManualTimer:
    """A stand-in for time.monotonic() that only advances when told to."""

    def __init__(self, start=0.0):
        self._now = start

    def __call__(self):
        return self._now

    def tick(self, amount=1.0):
        self._now += amount


@pytest.fixture
def timer():
    return ManualTimer()


# --- TTLCache ---


def test_ttl_basic_get_set(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    assert cache["a"] == 1
    assert "a" in cache


def test_ttl_expires_after_ttl(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    timer.tick(4)
    assert "a" in cache
    timer.tick(2)  # total elapsed = 6 > ttl 5
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_ttl_currsize_reflects_expiry(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    assert cache.currsize == 1
    timer.tick(10)
    assert cache.currsize == 0


def test_ttl_expire_method_returns_expired_pairs(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    timer.tick(10)
    expired = cache.expire()
    assert set(expired) == {("a", 1), ("b", 2)}
    assert len(cache) == 0


def test_ttl_lru_eviction_when_full(timer):
    cache = TTLCache(maxsize=2, ttl=100, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # touch "a"
    cache["c"] = 3  # should evict "b" (least recently used, not expired)
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_ttl_popitem_skips_still_valid_and_returns_lru(timer):
    cache = TTLCache(maxsize=10, ttl=100, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    key, value = cache.popitem()
    assert (key, value) == ("a", 1)


def test_ttl_iteration_excludes_expired(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    timer.tick(10)
    cache["b"] = 2
    assert list(iter(cache)) == ["b"]


def test_ttl_re_setting_key_refreshes_expiry(timer):
    cache = TTLCache(maxsize=10, ttl=5, timer=timer)
    cache["a"] = 1
    timer.tick(4)
    cache["a"] = 2  # refresh
    timer.tick(4)  # total 8 elapsed since first set, 4 since refresh
    assert cache["a"] == 2  # still valid: only 4 elapsed since refresh


# --- TLRUCache ---


def test_tlru_basic_get_set(timer):
    cache = TLRUCache(maxsize=10, ttu=lambda key, value, now: now + 5, timer=timer)
    cache["a"] = 1
    assert cache["a"] == 1


def test_tlru_expires_after_ttu(timer):
    cache = TLRUCache(maxsize=10, ttu=lambda key, value, now: now + 5, timer=timer)
    cache["a"] = 1
    timer.tick(10)
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_tlru_per_item_ttu(timer):
    # items with higher-priority values (larger ints) expire further out
    def ttu(key, value, now):
        return now + value

    cache = TLRUCache(maxsize=10, ttu=ttu, timer=timer)
    cache["short"] = 2
    cache["long"] = 20
    timer.tick(5)
    assert "short" not in cache
    assert "long" in cache


def test_tlru_setting_already_expired_item_is_a_no_op_insert(timer):
    # a ttu that returns a time in the past should mean the item is
    # effectively removed / never really available
    def ttu(key, value, now):
        return now - 1

    cache = TLRUCache(maxsize=10, ttu=ttu, timer=timer)
    cache["a"] = 1
    assert "a" not in cache


def test_tlru_expire_method_returns_expired_pairs(timer):
    cache = TLRUCache(maxsize=10, ttu=lambda k, v, now: now + 5, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    timer.tick(10)
    expired = cache.expire()
    assert set(expired) == {("a", 1), ("b", 2)}


def test_tlru_lru_eviction_when_full(timer):
    cache = TLRUCache(maxsize=2, ttu=lambda k, v, now: now + 100, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # touch "a"
    cache["c"] = 3  # should evict "b"
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


def test_tlru_popitem(timer):
    cache = TLRUCache(maxsize=10, ttu=lambda k, v, now: now + 100, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    key, value = cache.popitem()
    assert (key, value) == ("a", 1)


