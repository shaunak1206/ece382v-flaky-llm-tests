"""Tests for cachetools.func (lru_cache and friends) and cachetools.keys."""

import pickle
import threading
import time

import pytest

from cachetools import func, keys
from cachetools.keys import hashkey, methodkey, typedkey, typedmethodkey


class FakeTimer:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# ---------------------------------------------------------------------------
# cachetools.func
# ---------------------------------------------------------------------------


def test_lru_cache_bare_decorator_defaults_and_metadata():
    calls = []

    @func.lru_cache
    def f(x):
        """Docstring."""
        calls.append(x)
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]
    assert f.cache_parameters() == {"maxsize": 128, "typed": False}
    assert f.__name__ == "f"
    assert f.__doc__ == "Docstring."
    assert f.__wrapped__(1) == 2


def test_lru_cache_eviction_and_info():
    calls = []

    @func.lru_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)  # 2 is now least recently used
    f(3)  # evicts 2
    f(1)
    f(2)
    assert calls == [1, 2, 3, 2]
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (2, 4, 2, 2)
    assert type(info).__name__ == "CacheInfo"
    assert f.cache_parameters() == {"maxsize": 2, "typed": False}


def test_lru_cache_cache_clear_resets_stats():
    calls = []

    @func.lru_cache(maxsize=4)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    f.cache_clear()
    info = f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
    f(1)
    assert calls == [1, 1]


def test_lru_cache_unbounded_and_zero_sized():
    @func.lru_cache(maxsize=None)
    def unbounded(x):
        return x

    for i in range(500):
        unbounded(i)
    info = unbounded.cache_info()
    assert info.maxsize is None
    assert info.currsize == 500
    assert info.misses == 500
    assert unbounded.cache_parameters()["maxsize"] is None

    calls = []

    @func.lru_cache(maxsize=0)
    def nothing(x):
        calls.append(x)
        return x

    nothing(1)
    nothing(1)
    assert calls == [1, 1]
    info = nothing.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 2, 0)


def test_lru_cache_typed_versus_untyped():
    typed_calls = []

    @func.lru_cache(maxsize=10, typed=True)
    def typed(x):
        typed_calls.append(x)
        return x

    typed(1)
    typed(1.0)
    typed(1)
    assert len(typed_calls) == 2
    assert typed.cache_parameters() == {"maxsize": 10, "typed": True}

    untyped_calls = []

    @func.lru_cache(maxsize=10)
    def untyped(x):
        untyped_calls.append(x)
        return x

    untyped(1)
    untyped(1.0)
    assert len(untyped_calls) == 1


def test_lru_cache_keyword_arguments_and_exceptions():
    calls = []

    @func.lru_cache(maxsize=10)
    def f(a, b=0):
        calls.append((a, b))
        if a < 0:
            raise ValueError("negative")
        return a + b

    assert f(1, b=2) == 3
    assert f(1, b=2) == 3
    assert f(1, 2) == 3  # positional form is a different key
    assert len(calls) == 2
    with pytest.raises(ValueError):
        f(-1)
    with pytest.raises(ValueError):
        f(-1)
    assert calls.count((-1, 0)) == 2  # exceptions are never cached


def test_lru_cache_is_thread_safe():
    calls = []

    @func.lru_cache(maxsize=None)
    def slow(x):
        calls.append(x)
        time.sleep(0.1)
        return x * 3

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(slow(7))) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [21] * 6
    assert calls == [7]


def test_fifo_cache_and_lfu_cache_eviction():
    fifo_calls = []

    @func.fifo_cache(maxsize=2)
    def fifo(x):
        fifo_calls.append(x)
        return x

    fifo(1)
    fifo(2)
    fifo(1)  # hit, but does not change FIFO order
    fifo(3)  # evicts 1
    fifo(1)  # recomputed
    assert fifo_calls == [1, 2, 3, 1]

    lfu_calls = []

    @func.lfu_cache(maxsize=2)
    def lfu(x):
        lfu_calls.append(x)
        return x

    lfu(1)
    lfu(1)
    lfu(1)
    lfu(2)
    lfu(3)  # evicts 2, the least frequently used
    lfu(1)  # still cached
    lfu(2)  # recomputed
    assert lfu_calls == [1, 2, 3, 2]


def test_rr_cache_deterministic_choice_and_random_eviction():
    calls = []

    @func.rr_cache(maxsize=2, choice=lambda candidates: candidates[0])
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)  # evicts the first key (1)
    f(2)  # still cached
    f(1)  # recomputed
    assert calls == [1, 2, 3, 1]

    @func.rr_cache(maxsize=5)
    def g(x):
        return x

    for i in range(100):
        g(i)
    info = g.cache_info()
    assert info.currsize == 5
    assert info.maxsize == 5
    assert info.misses == 100


def test_ttl_cache_expiry_with_fake_timer():
    timer = FakeTimer()
    calls = []

    @func.ttl_cache(maxsize=10, ttl=10, timer=timer)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    timer.advance(9)
    f(1)
    assert len(calls) == 1
    timer.advance(2)
    f(1)
    assert len(calls) == 2
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize) == (2, 2, 10)


def test_ttl_cache_real_time_expiry():
    calls = []

    @func.ttl_cache(maxsize=10, ttl=0.2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert len(calls) == 1
    time.sleep(0.3)
    f(1)
    assert len(calls) == 2


def test_ttl_cache_bare_decorator_and_unbounded():
    @func.ttl_cache
    def bare(x):
        return x

    bare(1)
    assert bare.cache_parameters() == {"maxsize": 128, "typed": False}
    assert bare.cache_info().maxsize == 128

    timer = FakeTimer()

    @func.ttl_cache(maxsize=None, ttl=5, timer=timer)
    def unbounded(x):
        return x

    for i in range(300):
        unbounded(i)
    info = unbounded.cache_info()
    assert info.maxsize is None
    assert info.currsize == 300
    timer.advance(6)
    assert unbounded.cache_info().currsize == 0


# ---------------------------------------------------------------------------
# cachetools.keys
# ---------------------------------------------------------------------------


def test_hashkey_positional_arguments():
    assert hashkey() == ()
    assert hashkey(1, 2, 3) == (1, 2, 3)
    assert isinstance(hashkey(1, 2, 3), tuple)
    assert hashkey("a", 1) == hashkey("a", 1)
    assert hash(hashkey("a", 1)) == hash(hashkey("a", 1))
    assert hash(hashkey(1, 2)) == hash((1, 2))
    assert hashkey(1) == hashkey(1.0)
    assert {hashkey(1, 2): "v"}[hashkey(1, 2)] == "v"


def test_hashkey_keyword_arguments():
    assert hashkey(a=1, b=2) == hashkey(b=2, a=1)
    assert hash(hashkey(a=1, b=2)) == hash(hashkey(b=2, a=1))
    assert hashkey(a=1) != hashkey("a", 1)
    assert hashkey(1, a=2) != hashkey(1, 2)
    key = hashkey(1, a=2)
    assert key[0] == 1
    assert key[-1] == ("a", 2)
    assert len(key) == 3


def test_hashkey_hash_caching_concatenation_and_pickling():
    class Counting:
        hash_calls = 0

        def __hash__(self):
            Counting.hash_calls += 1
            return 42

    key = hashkey(Counting())
    hash(key)
    hash(key)
    hash(key)
    assert Counting.hash_calls == 1

    joined = hashkey(1) + (2,)
    assert joined == (1, 2)
    assert type(joined) is type(hashkey(1))
    prefixed = (0,) + hashkey(1)
    assert prefixed == (0, 1)
    assert type(prefixed) is type(hashkey(1))

    with pytest.raises(TypeError):
        hash(hashkey([1, 2]))

    original = hashkey(1, "two", z=3)
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert hash(restored) == hash(original)


def test_methodkey_ignores_self():
    class A:
        pass

    assert methodkey(A(), 1, 2) == methodkey(A(), 1, 2)
    assert methodkey(A(), 1, 2) == hashkey(1, 2)
    assert methodkey(object(), 1, b=2) == hashkey(1, b=2)
    assert methodkey(object(), 1, b=2) != methodkey(object(), 1, b=3)


def test_typedkey_and_typedmethodkey():
    assert typedkey(1) != typedkey(1.0)
    assert typedkey(1) == typedkey(1)
    assert typedkey(1, "a") == (1, "a", int, str)
    assert typedkey(1, a=2) != typedkey(1, a=2.0)
    assert typedkey(1, a=2) == typedkey(1, a=2)
    assert typedkey(a=1, b=2) == typedkey(b=2, a=1)
    assert hash(typedkey(1, a="x")) == hash(typedkey(1, a="x"))
    assert typedmethodkey(object(), 1) == typedkey(1)
    assert typedmethodkey(object(), 1) != typedmethodkey(object(), 1.0)
