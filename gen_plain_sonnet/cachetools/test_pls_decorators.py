import operator
import threading

import pytest

from cachetools import Cache, LRUCache, cached, cachedmethod, keys
from cachetools.func import fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache


class FakeTimer:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------
# cached
# ---------------------------------------------------------------------------


def test_cached_memoizes_and_cache_clear_forces_recompute():
    calls = []

    @cached(cache={})
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    assert f(1) == 1
    assert f(1) == 1
    assert f(1, y=2) == 3
    assert f(1, y=2) == 3
    assert calls == [(1, 0), (1, 2)]
    f.cache_clear()
    assert f(1) == 1
    assert len(calls) == 3


def test_cached_custom_key_function():
    calls = []

    @cached(cache={}, key=lambda x: x.lower())
    def f(x):
        calls.append(x)
        return x.upper()

    assert f("Abc") == "ABC"
    assert f("aBC") == "ABC"
    assert calls == ["Abc"]


def test_cached_with_none_cache_never_caches():
    calls = []

    @cached(cache=None, info=True)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    f(1)
    assert calls == [1, 1, 1]
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (0, 3, 0, 0)
    f.cache_clear()
    assert f.cache_info().misses == 0


def test_cached_info_reports_hits_misses_and_sizes():
    @cached(LRUCache(maxsize=2), info=True)
    def f(x):
        return x * 2

    f(1)
    f(1)
    f(2)
    f(3)
    info = f.cache_info()
    assert type(info).__name__ == "CacheInfo"
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 3, 2, 2)
    f.cache_clear()
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (0, 0, 2, 0)

    @cached({}, info=True)
    def g(x):
        return x

    g(1)
    g(2)
    g(1)
    info = g.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, None, 2)


def test_cached_exposes_wrapper_attributes():
    cache = {}
    lock = threading.Lock()
    cond = threading.Condition()

    @cached(cache, lock=lock)
    def f(x):
        """doc of f"""
        return x

    @cached(cache, condition=cond)
    def g(x):
        return x

    assert f.cache is cache
    assert f.cache_key is keys.hashkey
    assert f.cache_lock is lock
    assert f.cache_condition is None
    assert f.cache_info is None
    assert f.__name__ == "f"
    assert f.__doc__ == "doc of f"
    assert g.cache_lock is cond
    assert g.cache_condition is cond
    assert f(1) == 1
    assert g(2) == 2


_SHARED_LOCK = threading.RLock()


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"lock": threading.Lock()},
        {"condition": threading.Condition()},
        {"lock": _SHARED_LOCK, "condition": threading.Condition(_SHARED_LOCK)},
    ],
)
def test_cached_exception_results_are_not_cached(extra):
    calls = []

    @cached({}, **extra)
    def f(x):
        calls.append(x)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        f(1)
    with pytest.raises(RuntimeError):
        f(1)
    assert calls == [1, 1]


@pytest.mark.parametrize(
    "extra",
    [{}, {"lock": threading.Lock()}, {"condition": threading.Condition()}],
)
def test_cached_value_too_large_is_returned_but_not_stored(extra):
    cache = Cache(maxsize=1, getsizeof=len)
    calls = []

    @cached(cache, **extra)
    def f(x):
        calls.append(x)
        return "xx"

    assert f(1) == "xx"
    assert f(1) == "xx"
    assert calls == [1, 1]
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# cachedmethod
# ---------------------------------------------------------------------------


class Squarer:
    def __init__(self):
        self.cache = LRUCache(maxsize=4)
        self.lock = threading.RLock()
        self.calls = 0

    @cachedmethod(operator.attrgetter("cache"))
    def square(self, x):
        self.calls += 1
        return x * x

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def cube(self, x):
        self.calls += 1
        return x * x * x


def test_cachedmethod_uses_per_instance_cache_without_self_in_key():
    a = Squarer()
    b = Squarer()
    assert a.square(3) == 9
    assert a.square(3) == 9
    assert a.calls == 1
    assert b.calls == 0
    assert b.square(3) == 9
    assert b.calls == 1
    assert a.square.cache is a.cache
    assert keys.hashkey(3) in a.cache
    assert "square" in a.__dict__
    a.square.cache_clear()
    assert len(a.cache) == 0


def test_cachedmethod_with_lock_exposes_lock_and_caches():
    s = Squarer()
    assert s.cube(2) == 8
    assert s.cube(2) == 8
    assert s.calls == 1
    assert s.cube.cache_lock is s.lock
    assert s.cube.cache is s.cache


def test_cachedmethod_info_counts_and_clear():
    class Svc:
        def __init__(self):
            self.cache = LRUCache(maxsize=3)

        @cachedmethod(operator.attrgetter("cache"), info=True)
        def up(self, k):
            return k.upper()

    s = Svc()
    assert s.up("a") == "A"
    assert s.up("a") == "A"
    assert s.up("b") == "B"
    info = s.up.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, 3, 2)
    s.up.cache_clear()
    info = s.up.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (0, 0, 3, 0)


def test_cachedmethod_info_requires_instance_dict():
    class Slotted:
        __slots__ = ("cache",)

        def __init__(self):
            self.cache = {}

        @cachedmethod(operator.attrgetter("cache"), info=True)
        def f(self, x):
            return x

    with pytest.raises(TypeError):
        Slotted().f


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_hashkey_positional_and_keyword_arguments():
    assert keys.hashkey() == ()
    assert keys.hashkey(1, 2) == (1, 2)
    assert hash(keys.hashkey(1, 2)) == hash((1, 2))
    assert keys.hashkey(a=1, b=2) == keys.hashkey(b=2, a=1)
    assert keys.hashkey(1, a=2) != keys.hashkey(1)
    assert keys.hashkey(1, 2) == keys.hashkey(1, 2.0)


def test_typedkey_distinguishes_argument_types():
    assert keys.typedkey(1) == keys.typedkey(1)
    assert keys.typedkey(1) != keys.typedkey(1.0)
    assert keys.typedkey(a=1) != keys.typedkey(a=1.0)
    assert keys.typedkey(1) != keys.hashkey(1)
    d = {keys.typedkey(1): "int"}
    assert keys.typedkey(1.0) not in d
    assert d[keys.typedkey(1)] == "int"


def test_methodkey_variants_ignore_self():
    assert keys.methodkey(object(), 1, a=2) == keys.hashkey(1, a=2)
    assert keys.typedmethodkey(object(), 1, a=2) == keys.typedkey(1, a=2)


# ---------------------------------------------------------------------------
# cachetools.func
# ---------------------------------------------------------------------------


def test_func_lru_cache_eviction_info_and_parameters():
    calls = []

    @lru_cache(maxsize=2)
    def f(x):
        calls.append(x)
        return x * 10

    for arg in (1, 2, 1, 3, 2):
        assert f(arg) == arg * 10
    assert calls == [1, 2, 3, 2]
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 4, 2, 2)
    assert f.cache_parameters() == {"maxsize": 2, "typed": False}
    f.cache_clear()
    info = f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)


def test_func_lru_cache_bare_decorator_and_unbounded():
    @lru_cache
    def f(x):
        return x

    assert f.cache_parameters() == {"maxsize": 128, "typed": False}
    assert f(1) == 1
    assert f.cache_info().maxsize == 128

    @lru_cache(maxsize=None)
    def g(x):
        return x

    g(1)
    g(2)
    g(1)
    info = g.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, None, 2)
    assert g.cache_parameters() == {"maxsize": None, "typed": False}


def test_func_lru_cache_typed_flag():
    untyped_calls = []
    typed_calls = []

    @lru_cache(maxsize=8)
    def untyped(x):
        untyped_calls.append(x)
        return x

    @lru_cache(maxsize=8, typed=True)
    def typed(x):
        typed_calls.append(x)
        return x

    untyped(1)
    untyped(1.0)
    typed(1)
    typed(1.0)
    assert len(untyped_calls) == 1
    assert len(typed_calls) == 2


def test_func_ttl_cache_expires_with_custom_timer():
    timer = FakeTimer()
    calls = []

    @ttl_cache(maxsize=4, ttl=10, timer=timer)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    timer.now = 10
    f(1)
    assert calls == [1, 1]
    info = f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 2, 1)


@pytest.mark.parametrize(
    "make",
    [
        lambda: fifo_cache(maxsize=1),
        lambda: lfu_cache(maxsize=1),
        lambda: lru_cache(maxsize=1),
        lambda: rr_cache(maxsize=1),
    ],
)
def test_func_policy_decorators_respect_maxsize(make):
    calls = []

    @make()
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1]
    f(2)
    f(1)
    assert calls == [1, 2, 1]
    assert f.cache_info().currsize == 1
