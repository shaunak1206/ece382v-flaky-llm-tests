"""Tests for the @cached and @cachedmethod decorators."""

import operator
import threading
import time

import pytest

from cachetools import LRUCache, TTLCache, cached, cachedmethod, keys


class FakeTimer:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class CountingLock:
    """A reentrant lock that counts how many times it was entered."""

    def __init__(self):
        self.entered = 0
        self._lock = threading.RLock()

    def __enter__(self):
        self.entered += 1
        self._lock.acquire()
        return self

    def __exit__(self, *exc_info):
        self._lock.release()
        return False


# ---------------------------------------------------------------------------
# @cached
# ---------------------------------------------------------------------------


def test_cached_memoizes_by_arguments():
    calls = []

    @cached(cache={})
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(2, 1) == 3
    assert add(a=1, b=5) == 6
    assert add(b=5, a=1) == 6  # keyword order does not matter
    assert calls == [(1, 2), (2, 1), (1, 5)]


def test_cached_custom_and_typed_keys():
    calls = []

    @cached(cache={}, key=lambda x, y: x)
    def f(x, y):
        calls.append((x, y))
        return x + y

    assert f(1, 10) == 11
    assert f(1, 99) == 11  # y ignored by the key, cached value returned
    assert f(2, 5) == 7
    assert calls == [(1, 10), (2, 5)]

    typed_calls = []

    @cached(cache={}, key=keys.typedkey)
    def g(x):
        typed_calls.append(x)
        return x

    g(1)
    g(1.0)
    g(1)
    assert len(typed_calls) == 2

    untyped_calls = []

    @cached(cache={})
    def h(x):
        untyped_calls.append(x)
        return x

    h(1)
    h(1.0)
    assert len(untyped_calls) == 1


def test_cached_exceptions_are_not_cached_but_none_is():
    state = {"calls": 0}

    @cached(cache={})
    def flaky(x):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("first call fails")
        return None

    with pytest.raises(RuntimeError):
        flaky(1)
    assert flaky(1) is None
    assert flaky(1) is None
    assert state["calls"] == 2


def test_cached_with_lru_cache_evicts_old_entries():
    calls = []

    @cached(cache=LRUCache(maxsize=2))
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(3)  # evicts 1
    f(1)  # recomputed
    assert calls == [1, 2, 3, 1]


def test_cached_with_ttl_cache_expiry():
    timer = FakeTimer()
    calls = []

    @cached(cache=TTLCache(maxsize=10, ttl=10, timer=timer))
    def f(x):
        calls.append(x)
        return x * 2

    assert f(2) == 4
    timer.advance(5)
    assert f(2) == 4
    assert len(calls) == 1
    timer.advance(6)
    assert f(2) == 4
    assert len(calls) == 2


def test_cached_with_ttl_cache_real_time():
    calls = []

    @cached(cache=TTLCache(maxsize=10, ttl=0.2))
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert len(calls) == 1
    time.sleep(0.3)
    f(1)
    assert len(calls) == 2


def test_cached_info():
    @cached(cache={})
    def plain(x):
        return x

    assert plain.cache_info is None

    @cached(cache=LRUCache(maxsize=3), info=True)
    def bounded(x):
        return x

    bounded(1)
    bounded(1)
    bounded(2)
    info = bounded.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, 3, 2)
    assert type(info).__name__ == "CacheInfo"

    @cached(cache={}, info=True)
    def unbounded(x):
        return x

    unbounded(1)
    unbounded(2)
    unbounded(1)
    info = unbounded.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, None, 2)


def test_cached_cache_clear_resets_cache_and_counters():
    cache = LRUCache(maxsize=3)
    calls = []

    @cached(cache=cache, info=True)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    f.cache_clear()
    assert len(cache) == 0
    info = f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
    f(1)
    assert calls == [1, 1]

    plain = {}

    @cached(cache=plain)
    def g(x):
        return x

    g(1)
    assert plain
    g.cache_clear()
    assert not plain


def test_cached_exposes_attributes_and_metadata():
    cache = LRUCache(maxsize=2)
    lock = threading.Lock()

    def keyfunc(*args):
        return args

    @cached(cache=cache, key=keyfunc, lock=lock)
    def documented(x):
        """Return x unchanged."""
        return x

    assert documented.cache is cache
    assert documented.cache_key is keyfunc
    assert documented.cache_lock is lock
    assert documented.cache_condition is None
    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Return x unchanged."
    assert documented.__wrapped__(5) == 5

    store = {}

    @cached(cache=store)
    def f(x, y=1):
        return x + y

    assert f.cache_key is keys.hashkey
    assert f.cache_lock is None
    f(1, y=2)
    assert store[keys.hashkey(1, y=2)] == 3


def test_cached_with_lock_enters_lock():
    lock = CountingLock()
    calls = []

    @cached(cache={}, lock=lock)
    def f(x):
        calls.append(x)
        return x

    f(1)
    assert lock.entered == 2  # lookup + store
    f(1)
    assert lock.entered == 3  # lookup only
    assert calls == [1]


def test_cached_with_real_lock_condition_and_info():
    @cached(cache=LRUCache(maxsize=2), lock=threading.Lock(), info=True)
    def locked(x):
        return x * 2

    assert locked(2) == 4
    assert locked(2) == 4
    assert (locked.cache_info().hits, locked.cache_info().misses) == (1, 1)
    locked.cache_clear()
    assert locked.cache_info().misses == 0

    cond = threading.Condition()

    @cached(cache={}, condition=cond)
    def with_cond(x):
        return x

    assert with_cond.cache_lock is cond
    assert with_cond.cache_condition is cond

    rlock = threading.RLock()
    both_cond = threading.Condition(rlock)

    @cached(cache={}, lock=rlock, condition=both_cond, info=True)
    def with_both(x):
        return x

    with_both(1)
    with_both(1)
    assert with_both.cache_lock is rlock
    assert with_both.cache_condition is both_cond
    assert with_both.cache_info().hits == 1


def test_cached_condition_prevents_cache_stampede():
    calls = []
    results = []

    @cached(cache={}, condition=threading.Condition())
    def slow(x):
        calls.append(x)
        time.sleep(0.2)
        return x * 2

    def worker():
        results.append(slow(3))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [3]
    assert results == [6] * 5


def test_cached_with_none_cache_disables_caching():
    calls = []

    @cached(cache=None)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert calls == [1, 1]
    f.cache_clear()  # must not fail

    @cached(cache=None, info=True)
    def g(x):
        return x

    g(1)
    g(1)
    info = g.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (0, 2, 0, 0)
    g.cache_clear()
    assert g.cache_info().misses == 0


def test_cached_oversized_value_and_unhashable_argument():
    cache = LRUCache(maxsize=2, getsizeof=len)
    calls = []

    @cached(cache=cache)
    def f(x):
        calls.append(x)
        return "abc"

    assert f(1) == "abc"
    assert f(1) == "abc"
    assert len(calls) == 2
    assert len(cache) == 0

    @cached(cache={})
    def length(x):
        return len(x)

    with pytest.raises(TypeError):
        length([1, 2, 3])


# ---------------------------------------------------------------------------
# @cachedmethod
# ---------------------------------------------------------------------------


class Calc:
    def __init__(self, factor=1):
        self.cache = LRUCache(maxsize=4)
        self.factor = factor
        self.calls = 0

    @cachedmethod(operator.attrgetter("cache"))
    def scale(self, x):
        """Scale x."""
        self.calls += 1
        return x * self.factor


def test_cachedmethod_memoizes_per_instance():
    a = Calc(factor=2)
    b = Calc(factor=10)
    assert a.scale(5) == 10
    assert a.scale(5) == 10
    assert a.scale(x=5) == 10  # kwargs form is a separate key
    assert b.scale(5) == 50
    assert a.calls == 2
    assert b.calls == 1
    assert a.scale.cache is a.cache
    assert len(a.cache) == 2
    assert Calc.scale.__name__ == "scale"
    assert a.scale.__doc__ == "Scale x."


def test_cachedmethod_self_is_not_part_of_default_key():
    shared = LRUCache(maxsize=10)

    class Shared:
        def __init__(self, factor):
            self.factor = factor

        @cachedmethod(lambda self: shared)
        def scale(self, x):
            return x * self.factor

    a = Shared(2)
    b = Shared(100)
    assert a.scale(3) == 6
    # b hits the entry stored by a, because methodkey drops self
    assert b.scale(3) == 6


def test_cachedmethod_custom_and_typed_keys():
    class Keyed:
        def __init__(self):
            self.cache = {}
            self.typed_cache = {}
            self.calls = 0

        @cachedmethod(operator.attrgetter("cache"), key=lambda self, x, y: x)
        def f(self, x, y):
            self.calls += 1
            return x + y

        @cachedmethod(operator.attrgetter("typed_cache"), key=keys.typedmethodkey)
        def g(self, x):
            self.calls += 1
            return x

    obj = Keyed()
    assert obj.f(1, 2) == 3
    assert obj.f(1, 50) == 3
    assert obj.calls == 1
    obj.g(1)
    obj.g(1.0)
    obj.g(1)
    assert obj.calls == 3


def test_cachedmethod_cache_clear_and_exceptions_not_cached():
    calc = Calc()
    calc.scale(1)
    calc.scale(2)
    calc.scale.cache_clear()
    assert len(calc.cache) == 0
    calc.scale(1)
    assert calc.calls == 3

    class Flaky:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod(operator.attrgetter("cache"))
        def f(self, x):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("boom")
            return x

    obj = Flaky()
    with pytest.raises(ValueError):
        obj.f(1)
    assert obj.f(1) == 1
    assert obj.f(1) == 1
    assert obj.calls == 2


def test_cachedmethod_with_lock():
    class Locked:
        def __init__(self):
            self.cache = {}
            self.lock = CountingLock()

        @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
        def f(self, x):
            return x

    obj = Locked()
    obj.f(1)
    assert obj.lock.entered == 2
    obj.f(1)
    assert obj.lock.entered == 3
    assert obj.f.cache_lock is obj.lock


def test_cachedmethod_info():
    class Informed:
        def __init__(self):
            self.cache = LRUCache(maxsize=5)
            self.dict_cache = {}
            self.lock = threading.RLock()

        @cachedmethod(operator.attrgetter("cache"), info=True)
        def f(self, x):
            return x

        @cachedmethod(
            operator.attrgetter("dict_cache"),
            lock=operator.attrgetter("lock"),
            info=True,
        )
        def g(self, x):
            return x

    obj = Informed()
    obj.f(1)
    obj.f(1)
    obj.f(2)
    info = obj.f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 2, 5, 2)
    obj.f.cache_clear()
    info = obj.f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)

    obj.g(1)
    obj.g(1)
    info = obj.g.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 1, None, 1)


def test_cachedmethod_condition_prevents_stampede():
    class Slow:
        def __init__(self):
            self.cache = {}
            self.cond = threading.Condition()
            self.calls = 0

        @cachedmethod(
            operator.attrgetter("cache"), condition=operator.attrgetter("cond")
        )
        def f(self, x):
            self.calls += 1
            time.sleep(0.2)
            return x * 2

    obj = Slow()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(obj.f(4))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert obj.calls == 1
    assert results == [8] * 4


def test_cachedmethod_with_ttl_cache_expiry():
    timer = FakeTimer()

    class Timed:
        def __init__(self):
            self.cache = TTLCache(maxsize=5, ttl=10, timer=timer)
            self.calls = 0

        @cachedmethod(operator.attrgetter("cache"))
        def f(self, x):
            self.calls += 1
            return x

    obj = Timed()
    obj.f(1)
    timer.advance(5)
    obj.f(1)
    assert obj.calls == 1
    timer.advance(6)
    obj.f(1)
    assert obj.calls == 2
