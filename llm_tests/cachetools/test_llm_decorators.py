"""Tests for the @cached and @cachedmethod decorators."""

import threading

import pytest

from cachetools import LRUCache, TTLCache, cached, cachedmethod, keys


# --- @cached basics ---


def test_cached_caches_result():
    cache = LRUCache(maxsize=10)
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]  # only called once


def test_cached_different_args_not_shared():
    cache = LRUCache(maxsize=10)
    calls = []

    @cached(cache)
    def f(x):
        calls.append(x)
        return x * 2

    f(1)
    f(2)
    assert calls == [1, 2]


def test_cached_populates_cache_object():
    cache = LRUCache(maxsize=10)

    @cached(cache)
    def f(x):
        return x * 2

    f(3)
    assert cache[keys.hashkey(3)] == 6


def test_cached_cache_none_disables_caching():
    calls = []

    @cached(cache=None)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1, 1]  # called every time, no caching


def test_cached_custom_key_function():
    cache = LRUCache(maxsize=10)
    calls = []

    def ignore_case_key(s):
        return keys.hashkey(s.lower())

    @cached(cache, key=ignore_case_key)
    def f(s):
        calls.append(s)
        return s.upper()

    assert f("Hello") == "HELLO"
    assert f("HELLO") == "HELLO"
    assert calls == ["Hello"]  # second call was a cache hit


def test_cached_value_too_large_is_not_cached_but_still_returns():
    cache = LRUCache(maxsize=1, getsizeof=lambda v: 5)  # every value "too large"

    @cached(cache)
    def f(x):
        return x

    result = f(1)
    assert result == 1
    assert len(cache) == 0  # never actually stored


# --- @cached with lock ---


def test_cached_with_lock_concurrent_calls_thread_safe():
    cache = LRUCache(maxsize=10)
    lock = threading.Lock()
    call_count = [0]

    @cached(cache, lock=lock)
    def f(x):
        call_count[0] += 1
        return x

    threads = [threading.Thread(target=f, args=(1,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert f(1) == 1
    # cache should contain exactly one result for key 1
    assert cache[keys.hashkey(1)] == 1


# --- @cached info / cache_info / cache_clear ---


def test_cached_info_tracks_hits_and_misses():
    cache = LRUCache(maxsize=10)

    @cached(cache, info=True)
    def f(x):
        return x

    f(1)  # miss
    f(1)  # hit
    f(2)  # miss
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.currsize == 2
    assert info.maxsize == 10


def test_cached_cache_clear_resets_cache_and_stats():
    cache = LRUCache(maxsize=10)

    @cached(cache, info=True)
    def f(x):
        return x

    f(1)
    f(1)
    f.cache_clear()
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert len(cache) == 0


def test_cached_info_with_cache_none():
    @cached(cache=None, info=True)
    def f(x):
        return x

    f(1)
    f(1)
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 2
    assert info.maxsize == 0
    assert info.currsize == 0


# --- @cachedmethod ---


def test_cachedmethod_basic():
    class Obj:
        def __init__(self):
            self.cache = LRUCache(maxsize=10)
            self.calls = []

        @cachedmethod(lambda self: self.cache)
        def compute(self, x):
            self.calls.append(x)
            return x * 2

    obj = Obj()
    assert obj.compute(2) == 4
    assert obj.compute(2) == 4
    assert obj.calls == [2]


def test_cachedmethod_separate_instances_separate_caches():
    class Obj:
        def __init__(self):
            self.cache = LRUCache(maxsize=10)
            self.calls = []

        @cachedmethod(lambda self: self.cache)
        def compute(self, x):
            self.calls.append(x)
            return x * 2

    a = Obj()
    b = Obj()
    a.compute(1)
    b.compute(1)
    assert a.calls == [1]
    assert b.calls == [1]  # not shared with a's cache


def test_cachedmethod_cache_clear():
    class Obj:
        def __init__(self):
            self.cache = LRUCache(maxsize=10)

        @cachedmethod(lambda self: self.cache)
        def compute(self, x):
            return x

    obj = Obj()
    obj.compute(1)
    assert len(obj.cache) == 1
    obj.compute.cache_clear()
    assert len(obj.cache) == 0


def test_cachedmethod_with_custom_key():
    class Obj:
        def __init__(self):
            self.cache = LRUCache(maxsize=10)
            self.calls = []

        @cachedmethod(lambda self: self.cache, key=keys.methodkey)
        def compute(self, x, y):
            self.calls.append((x, y))
            return x + y

    obj = Obj()
    assert obj.compute(1, 2) == 3
    assert obj.compute(1, 2) == 3
    assert obj.calls == [(1, 2)]


def test_cachedmethod_info():
    class Obj:
        def __init__(self):
            self.cache = LRUCache(maxsize=10)

        @cachedmethod(lambda self: self.cache, info=True)
        def compute(self, x):
            return x

    obj = Obj()
    obj.compute(1)
    obj.compute(1)
    info = obj.compute.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_cachedmethod_ttlcache_backed():
    class ManualTimer:
        def __init__(self):
            self._now = 0.0

        def __call__(self):
            return self._now

        def tick(self, amount):
            self._now += amount

    timer = ManualTimer()

    class Obj:
        def __init__(self):
            self.cache = TTLCache(maxsize=10, ttl=5, timer=timer)
            self.calls = []

        @cachedmethod(lambda self: self.cache)
        def compute(self, x):
            self.calls.append(x)
            return x * 2

    obj = Obj()
    assert obj.compute(1) == 2
    assert obj.compute(1) == 2
    assert obj.calls == [1]
    timer.tick(10)
    assert obj.compute(1) == 2  # expired, recomputed
    assert obj.calls == [1, 1]
