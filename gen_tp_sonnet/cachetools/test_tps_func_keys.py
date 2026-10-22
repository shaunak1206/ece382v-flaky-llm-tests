import pytest

import cachetools
import cachetools.func
from cachetools import keys


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


# ------------------------------------------------------------ func module

def test_func_lru_cache_info_counts_hits_and_misses():
    @cachetools.func.lru_cache(maxsize=2)
    def f(x):
        return x * 10

    f(1)
    f(1)
    f(2)
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 2
    assert info.currsize == 2


def test_func_lru_cache_evicts_least_recently_used():
    calls = []

    @cachetools.func.lru_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)
    f(3)
    f(1)
    assert calls == [1, 2, 3]
    f(2)
    assert calls == [1, 2, 3, 2]


def test_func_cache_clear_forces_recompute():
    calls = []

    @cachetools.func.lru_cache(maxsize=4)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f.cache_clear()
    assert f.cache_info().currsize == 0
    f(1)
    assert calls == [1, 1]


def test_func_cache_parameters():
    @cachetools.func.lru_cache(maxsize=7, typed=True)
    def f(x):
        return x

    params = f.cache_parameters()
    assert params["maxsize"] == 7
    assert params["typed"] is True


def test_func_fifo_cache_evicts_oldest_regardless_of_hits():
    calls = []

    @cachetools.func.fifo_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)
    f(3)
    assert calls == [1, 2, 3]
    f(1)
    assert calls == [1, 2, 3, 1]


def test_func_lfu_cache_keeps_frequently_used_entries():
    calls = []

    @cachetools.func.lfu_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    f(1)
    f(2)
    f(3)
    f(1)
    assert calls == [1, 2, 3]


def test_func_rr_cache_with_deterministic_choice():
    calls = []

    @cachetools.func.rr_cache(maxsize=2, choice=min)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)
    f(2)
    f(3)
    assert calls == [1, 2, 3]
    f(1)
    assert calls == [1, 2, 3, 1]


def test_func_ttl_cache_expires_results():
    clock = Clock()
    calls = []

    @cachetools.func.ttl_cache(maxsize=4, ttl=10, timer=clock)
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


def test_func_typed_flag_separates_int_and_float():
    typed_calls = []
    untyped_calls = []

    @cachetools.func.lru_cache(maxsize=4, typed=True)
    def typed(x):
        typed_calls.append(x)
        return x

    @cachetools.func.lru_cache(maxsize=4, typed=False)
    def untyped(x):
        untyped_calls.append(x)
        return x

    typed(1)
    typed(1.0)
    untyped(1)
    untyped(1.0)
    assert len(typed_calls) == 2
    assert len(untyped_calls) == 1


# ------------------------------------------------------------ keys module

def test_hashkey_equal_for_same_arguments():
    assert keys.hashkey(1, 2) == keys.hashkey(1, 2)
    assert hash(keys.hashkey(1, 2)) == hash(keys.hashkey(1, 2))
    assert keys.hashkey("a", x=1) == keys.hashkey("a", x=1)


def test_hashkey_positional_order_matters():
    assert keys.hashkey(1, 2) != keys.hashkey(2, 1)
    assert keys.hashkey() != keys.hashkey(1)


def test_hashkey_keyword_order_does_not_matter():
    assert keys.hashkey(a=1, b=2) == keys.hashkey(b=2, a=1)
    assert keys.hashkey(a=1, b=2) != keys.hashkey(a=1, b=3)


def test_hashkey_equal_int_and_float_collide():
    assert keys.hashkey(1) == keys.hashkey(1.0)


def test_typedkey_distinguishes_types():
    assert keys.typedkey(1) != keys.typedkey(1.0)
    assert keys.typedkey(1) == keys.typedkey(1)
    assert keys.typedkey("a", b=1) != keys.typedkey("a", b=1.0)


def test_hashkey_with_unhashable_argument_raises_typeerror():
    with pytest.raises(TypeError):
        hash(keys.hashkey([1, 2]))


def test_methodkey_ignores_self():
    class A:
        pass

    k1 = keys.methodkey(A(), 1, 2)
    k2 = keys.methodkey(A(), 1, 2)
    assert k1 == k2
    assert k1 == keys.hashkey(1, 2)
    assert keys.methodkey(A(), 1, 2) != keys.methodkey(A(), 2, 1)
