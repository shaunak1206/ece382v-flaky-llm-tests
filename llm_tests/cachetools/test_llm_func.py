"""Tests for cachetools.func: lru_cache, lfu_cache, fifo_cache, rr_cache,
ttl_cache -- the functools.lru_cache-compatible decorators.
"""

import time

import pytest

from cachetools.func import fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache


DECORATORS = [fifo_cache, lfu_cache, lru_cache, rr_cache]


@pytest.fixture(params=DECORATORS)
def decorator(request):
    return request.param


def test_basic_caching(decorator):
    calls = []

    @decorator(maxsize=10)
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls == [2]


def test_used_as_bare_decorator_no_parens(decorator):
    calls = []

    @decorator
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]


def test_maxsize_none_unbounded(decorator):
    calls = []

    @decorator(maxsize=None)
    def f(x):
        calls.append(x)
        return x

    for i in range(50):
        f(i)
    for i in range(50):
        f(i)
    assert calls == list(range(50))  # each computed exactly once


def test_cache_info_hits_and_misses(decorator):
    @decorator(maxsize=10)
    def f(x):
        return x

    f(1)
    f(1)
    f(2)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.currsize == 2
    assert info.maxsize == 10


def test_cache_clear_resets(decorator):
    @decorator(maxsize=10)
    def f(x):
        return x

    f(1)
    f.cache_clear()
    info = f.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0


def test_typed_distinguishes_arg_types(decorator):
    calls = []

    @decorator(maxsize=10, typed=True)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1.0)
    assert calls == [1, 1.0]  # treated as distinct keys


def test_untyped_does_not_distinguish_arg_types(decorator):
    calls = []

    @decorator(maxsize=10, typed=False)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1.0)
    assert calls == [1]  # 1 == 1.0 -> same cache key


def test_cache_parameters(decorator):
    @decorator(maxsize=7, typed=True)
    def f(x):
        return x

    params = f.cache_parameters()
    assert params == {"maxsize": 7, "typed": True}


# --- ttl_cache specific ---


def test_ttl_cache_basic_caching():
    calls = []

    @ttl_cache(maxsize=10, ttl=60)
    def f(x):
        calls.append(x)
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == [1]


def test_ttl_cache_expiry_with_manual_timer():
    class ManualTimer:
        def __init__(self):
            self._now = 0.0

        def __call__(self):
            return self._now

        def tick(self, amount):
            self._now += amount

    timer = ManualTimer()
    calls = []

    @ttl_cache(maxsize=10, ttl=5, timer=timer)
    def f(x):
        calls.append(x)
        return x

    f(1)
    timer.tick(10)
    f(1)
    assert calls == [1, 1]  # recomputed after expiry


def test_ttl_cache_maxsize_none():
    calls = []

    @ttl_cache(maxsize=None, ttl=60)
    def f(x):
        calls.append(x)
        return x

    for i in range(20):
        f(i)
    for i in range(20):
        f(i)
    assert calls == list(range(20))


def test_ttl_cache_cache_info():
    @ttl_cache(maxsize=10, ttl=60)
    def f(x):
        return x

    f(1)
    f(1)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_rr_cache_custom_choice():
    calls = []

    @rr_cache(maxsize=2, choice=lambda seq: seq[0])
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)  # eviction triggered with deterministic choice
    info = f.cache_info()
    assert info.currsize == 2
