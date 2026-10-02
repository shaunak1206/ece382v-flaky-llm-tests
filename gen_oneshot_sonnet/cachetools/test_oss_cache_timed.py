"""Tests for the time aware caches: TTLCache and TLRUCache."""

import time

import pytest

from cachetools import TLRUCache, TTLCache


class FakeTimer:
    """A controllable clock."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def timer():
    return FakeTimer()


# ---------------------------------------------------------------------------
# TTLCache with a fake clock
# ---------------------------------------------------------------------------


def test_ttl_properties(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    assert cache.ttl == 10
    assert cache.maxsize == 5
    assert cache.currsize == 0
    assert cache.timer() == 0
    timer.advance(3)
    assert cache.timer() == 3


def test_ttl_item_lifetime(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    timer.advance(9.5)
    assert "a" in cache
    assert cache["a"] == 1
    timer.advance(0.5)  # exactly at the ttl boundary
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]


def test_ttl_len_and_currsize_ignore_expired_items(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    timer.advance(6)
    cache["b"] = 2
    assert len(cache) == 2
    assert cache.currsize == 2
    timer.advance(5)
    assert len(cache) == 1
    timer.advance(10)
    assert cache.currsize == 0


def test_ttl_get_pop_and_delete_on_expired_items(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    timer.advance(10)
    assert cache.get("a") is None
    assert cache.get("a", "dflt") == "dflt"
    assert cache.pop("b", "dflt") == "dflt"
    with pytest.raises(KeyError):
        cache.pop("b")
    with pytest.raises(KeyError):
        del cache["c"]


def test_ttl_overwrite_resets_expiry_but_reads_do_not(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    cache["r"] = 1
    timer.advance(5)
    cache["a"] = 2  # rewrite refreshes the ttl
    assert cache["r"] == 1  # read does not
    timer.advance(7)  # t=12
    assert cache["a"] == 2
    assert "r" not in cache
    timer.advance(3)  # t=15
    assert "a" not in cache


def test_ttl_expire_returns_expired_pairs(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    timer.advance(5)
    cache["b"] = 2
    timer.advance(6)
    assert cache.expire() == [("a", 1)]
    assert list(cache) == ["b"]
    assert cache.expire(time=5) == []
    assert cache.expire(time=100) == [("b", 2)]


def test_ttl_eviction_prefers_expired_then_least_recently_used(timer):
    cache = TTLCache(maxsize=2, ttl=10, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]  # a becomes most recently used
    cache["c"] = 3
    assert "b" not in cache
    assert set(cache) == {"a", "c"}

    timer.advance(10)
    cache["d"] = 4
    timer.advance(5)
    cache["e"] = 5  # a and c expired; d is still alive
    assert set(cache) == {"d", "e"}


def test_ttl_popitem_skips_expired_and_empty_raises(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    cache["a"] = 1
    timer.advance(5)
    cache["b"] = 2
    timer.advance(6)
    assert list(cache) == ["b"]  # iteration skips the expired entry
    assert list(cache.keys()) == ["b"]
    assert cache.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        cache.popitem()


def test_ttl_setdefault_and_clear(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer)
    assert cache.setdefault("a", 1) == 1
    assert cache.setdefault("a", 2) == 1
    timer.advance(10)
    assert cache.setdefault("a", 3) == 3
    cache["b"] = 4
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0
    cache["c"] = 5
    assert list(cache) == ["c"]


def test_ttl_expiry_frees_space_for_sized_values(timer):
    cache = TTLCache(maxsize=5, ttl=10, timer=timer, getsizeof=len)
    cache["a"] = "aaaa"
    timer.advance(11)
    cache["b"] = "bbbbb"
    assert cache.currsize == 5
    assert list(cache) == ["b"]
    with pytest.raises(ValueError):
        cache["c"] = "toolarge"


# ---------------------------------------------------------------------------
# TTLCache with the real clock
# ---------------------------------------------------------------------------


def test_ttl_real_time_expiry():
    cache = TTLCache(maxsize=10, ttl=0.2)
    cache["a"] = 1
    assert cache["a"] == 1
    time.sleep(0.3)
    assert "a" not in cache
    with pytest.raises(KeyError):
        cache["a"]

    # items inserted at different moments expire at different moments
    cache = TTLCache(maxsize=10, ttl=0.2)
    cache["old"] = 1
    time.sleep(0.15)
    cache["new"] = 2
    time.sleep(0.1)
    assert "old" not in cache
    assert "new" in cache
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# TLRUCache with a fake clock
# ---------------------------------------------------------------------------


def ttu_value_as_ttl(_key, value, now):
    return now + value


def ttu_const_ten(_key, _value, now):
    return now + 10


def test_tlru_properties_and_ttu_arguments(timer):
    seen = []

    def ttu(key, value, now):
        seen.append((key, value, now))
        return now + 5

    cache = TLRUCache(maxsize=4, ttu=ttu, timer=timer)
    assert cache.ttu is ttu
    assert cache.maxsize == 4
    assert cache.currsize == 0
    timer.advance(3)
    cache["k"] = "v"
    assert seen == [("k", "v", 3)]


def test_tlru_per_item_expiry(timer):
    cache = TLRUCache(maxsize=10, ttu=ttu_value_as_ttl, timer=timer)
    cache["short"] = 5
    cache["long"] = 10
    timer.advance(6)
    assert "short" not in cache
    assert cache["long"] == 10
    assert len(cache) == 1
    timer.advance(5)
    assert "long" not in cache
    assert len(cache) == 0


def test_tlru_already_expired_values_are_not_stored(timer):
    never = TLRUCache(maxsize=10, ttu=lambda k, v, now: now, timer=timer)
    never["a"] = 1
    assert "a" not in never
    assert len(never) == 0

    cache = TLRUCache(maxsize=10, ttu=ttu_value_as_ttl, timer=timer)
    cache["a"] = 10
    assert "a" in cache
    cache["a"] = 0  # replacing with an expired value removes the old one
    assert "a" not in cache
    assert len(cache) == 0


def test_tlru_getitem_on_expired_and_overwrite(timer):
    cache = TLRUCache(maxsize=10, ttu=ttu_const_ten, timer=timer)
    cache["a"] = 1
    timer.advance(7)
    cache["a"] = 2
    timer.advance(7)
    assert cache["a"] == 2
    timer.advance(3)
    with pytest.raises(KeyError):
        cache["a"]


def test_tlru_lru_eviction_and_expire_pairs(timer):
    cache = TLRUCache(maxsize=2, ttu=ttu_const_ten, timer=timer)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"]
    cache["c"] = 3
    assert "b" not in cache
    assert set(cache) == {"a", "c"}

    timed = TLRUCache(maxsize=10, ttu=ttu_value_as_ttl, timer=timer)
    timed["a"] = 5
    timed["b"] = 8
    timed["c"] = 20
    timer.advance(9)
    assert sorted(timed.expire()) == [("a", 5), ("b", 8)]
    assert list(timed) == ["c"]


def test_tlru_repeated_overwrites_keep_single_entry(timer):
    cache = TLRUCache(maxsize=10, ttu=ttu_const_ten, timer=timer)
    for i in range(200):
        cache["k"] = i
        timer.advance(0.01)
    assert len(cache) == 1
    assert cache["k"] == 199


def test_tlru_popitem_and_delete_with_expired_items(timer):
    cache = TLRUCache(maxsize=10, ttu=ttu_value_as_ttl, timer=timer)
    cache["a"] = 3
    cache["b"] = 30
    cache["c"] = 3
    timer.advance(5)
    with pytest.raises(KeyError):
        del cache["c"]
    assert cache.popitem() == ("b", 30)
    with pytest.raises(KeyError):
        cache.popitem()


def test_tlru_get_setdefault_and_clear(timer):
    cache = TLRUCache(maxsize=10, ttu=ttu_const_ten, timer=timer)
    assert cache.get("a") is None
    assert cache.setdefault("a", 1) == 1
    assert cache.get("a") == 1
    timer.advance(10)
    assert cache.get("a", "gone") == "gone"
    cache["b"] = 2
    cache.clear()
    assert len(cache) == 0
    cache["c"] = 3
    assert list(cache) == ["c"]


# ---------------------------------------------------------------------------
# TLRUCache with the real clock
# ---------------------------------------------------------------------------


def test_tlru_real_time_expiry():
    cache = TLRUCache(maxsize=10, ttu=lambda k, v, now: now + v)
    cache["short"] = 0.2
    cache["long"] = 5
    assert cache["short"] == 0.2
    time.sleep(0.3)
    assert "short" not in cache
    assert "long" in cache
    assert len(cache) == 1
