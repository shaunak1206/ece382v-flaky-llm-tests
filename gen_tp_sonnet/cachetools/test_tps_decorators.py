import operator

import pytest

from cachetools import LRUCache, TTLCache, cached, cachedmethod, keys


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class CountingLock:
    def __init__(self):
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        return False


def test_cached_memoizes_with_plain_dict():
    calls = []

    @cached(cache={})
    def square(x):
        calls.append(x)
        return x * x

    assert square(4) == 16
    assert square(4) == 16
    assert calls == [4]


def test_cached_distinguishes_positional_and_keyword_arguments():
    calls = []

    @cached(cache={})
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(2, 1) == 3
    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert add(1, 3) == 4
    assert len(calls) == 4


def test_cached_exposes_cache_attribute():
    cache = LRUCache(maxsize=4)

    @cached(cache)
    def f(x):
        return x + 1

    f(1)
    assert f.cache is cache
    assert 2 in cache.values()


def test_cached_does_not_store_exceptions():
    calls = []

    @cached(cache={})
    def boom(x):
        calls.append(x)
        raise ValueError("bad")

    for _ in range(2):
        with pytest.raises(ValueError):
            boom(1)
    assert len(calls) == 2


def test_cached_custom_key_function():
    calls = []

    @cached(cache={}, key=lambda x, y: x)
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 2) == 3
    assert f(1, 100) == 3
    assert f(2, 5) == 7
    assert calls == [(1, 2), (2, 5)]


def test_cached_typedkey_separates_int_and_float():
    calls = []

    @cached(cache={}, key=keys.typedkey)
    def ident(x):
        calls.append(x)
        return x

    ident(1)
    ident(1.0)
    ident(1)
    assert len(calls) == 2


def test_cached_default_key_treats_equal_int_and_float_alike():
    calls = []

    @cached(cache={})
    def ident(x):
        calls.append(x)
        return x

    ident(1)
    ident(1.0)
    assert len(calls) == 1


def test_cached_uses_supplied_lock():
    lock = CountingLock()

    @cached(cache={}, lock=lock)
    def f(x):
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert lock.enters >= 1


def test_cached_with_lru_cache_evicts_and_recomputes():
    calls = []

    @cached(cache=LRUCache(maxsize=2))
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)
    f(1)
    assert calls == [1, 2, 3, 1]


def test_cached_info_reports_hits_misses_and_clears():
    @cached(cache=LRUCache(maxsize=4), info=True)
    def f(x):
        return x

    f(1)
    f(1)
    f(2)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 4
    assert info.currsize == 2
    f.cache_clear()
    assert f.cache_info().currsize == 0


def test_cached_preserves_function_metadata():
    @cached(cache={})
    def documented(x):
        """Docstring here."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring here."
    assert documented.__wrapped__(5) == 5


def test_cached_with_ttlcache_recomputes_after_expiry():
    clock = Clock()
    calls = []

    @cached(cache=TTLCache(maxsize=4, ttl=10, timer=clock))
    def f(x):
        calls.append(x)
        return x

    f(1)
    clock.advance(5)
    f(1)
    assert len(calls) == 1
    clock.advance(10)
    f(1)
    assert len(calls) == 2


def test_cachedmethod_keeps_separate_cache_per_instance():
    class Foo:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod(operator.attrgetter("cache"))
        def double(self, x):
            self.calls += 1
            return x * 2

    a, b = Foo(), Foo()
    assert a.double(2) == 4
    assert b.double(2) == 4
    assert a.double(2) == 4
    assert a.calls == 1
    assert b.calls == 1


def test_cachedmethod_with_shared_cache_ignores_self_in_key():
    shared = {}

    class Foo:
        def __init__(self):
            self.calls = 0

        @cachedmethod(lambda self: shared)
        def triple(self, x):
            self.calls += 1
            return x * 3

    a, b = Foo(), Foo()
    assert a.triple(2) == 6
    assert b.triple(2) == 6
    assert a.calls == 1
    assert b.calls == 0
