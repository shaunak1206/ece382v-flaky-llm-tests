import threading
import warnings
from collections import OrderedDict

import pytest
from freezegun import freeze_time

import cachetools
from cachetools import (
    Cache,
    FIFOCache,
    LFUCache,
    LRUCache,
    RRCache,
    TLRUCache,
    TTLCache,
    cached,
    cachedmethod,
    keys,
)
from cachetools.func import fifo_cache, lfu_cache, lru_cache, rr_cache, ttl_cache


def test_version_and_all_exports():
    assert cachetools.__version__ == "7.1.8"
    for name in cachetools.__all__:
        assert hasattr(cachetools, name)


def test_cache_negative_maxsize_raises():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_basic_set_get_len_contains_iter_repr():
    c = Cache(3)
    c["a"] = 1
    c["b"] = 2
    assert len(c) == 2
    assert "a" in c and "x" not in c
    assert c["a"] == 1
    assert list(iter(c)) == ["a", "b"]
    r = repr(c)
    assert "Cache(" in r and "maxsize=3" in r


def test_cache_pop_and_get_default():
    c = Cache(2)
    c["a"] = 1
    assert c.pop("a") == 1
    assert c.get("missing", 9) == 9
    assert c.pop("missing", 7) == 7


def test_cache_pop_missing_raises_keyerror():
    c = Cache(1)
    with pytest.raises(KeyError):
        c.pop("x")


def test_cache_setdefault_stores_default():
    c = Cache(2)
    assert c.setdefault("a", 10) == 10
    assert c["a"] == 10
    assert c.setdefault("a", 20) == 10


def test_cache_clear_resets_currsize():
    c = Cache(2)
    c["a"] = 1
    c["b"] = 2
    assert c.currsize == 2
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0


def test_cache_custom_getsizeof_tracks_size():
    c = Cache(5, getsizeof=len)
    c["a"] = "xx"
    c["b"] = "yyy"
    assert c.currsize == 5
    assert c["a"] == "xx"
    assert c["b"] == "yyy"


def test_cache_setitem_negative_size_raises():
    c = Cache(5, getsizeof=lambda v: -1)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_setitem_value_too_large_raises():
    c = Cache(1, getsizeof=lambda v: 2)
    with pytest.raises(ValueError):
        c["a"] = 1


def test_cache_eviction_with_custom_size():
    c = Cache(3, getsizeof=len)
    c["a"] = "xx"
    c["b"] = "y"
    c["c"] = "z"
    assert c.currsize == 3
    c["d"] = "qq"
    assert len(c) == 2
    assert c.currsize == 3


def test_fifo_cache_evicts_in_insertion_order():
    c = FIFOCache(2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert list(c) == ["b", "c"]


def test_fifo_cache_popitem_returns_oldest():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert list(c) == ["b"]


def test_fifo_cache_reinserting_existing_moves_order_to_end():
    c = FIFOCache(3)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10
    assert c.popitem() == ("b", 2)


def test_fifo_cache_clear_empties_order():
    c = FIFOCache(2)
    c["a"] = 1
    c.clear()
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_cache_access_refreshes_order():
    c = LRUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "b" not in c
    assert "a" in c and "c" in c


def test_lru_cache_popitem_returns_least_recently_used():
    c = LRUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    _ = c["b"]
    assert c.popitem() == ("a", 1)


def test_lru_cache_clear_empties_order():
    c = LRUCache(2)
    c["a"] = 1
    c.clear()
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_cache_access_increases_frequency():
    c = LFUCache(2)
    c["a"] = 1
    c["b"] = 2
    _ = c["a"]
    c["c"] = 3
    assert "b" not in c
    assert "a" in c and "c" in c


def test_lfu_cache_popitem_returns_least_frequently_used():
    c = LFUCache(3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    _ = c["a"]
    _ = c["a"]
    _ = c["b"]
    key, value = c.popitem()
    assert (key, value) == ("c", 3)


def test_lfu_cache_clear_empties_and_popitem_raises():
    c = LFUCache(1)
    c["a"] = 1
    c.clear()
    with pytest.raises(KeyError):
        c.popitem()


def test_rr_cache_choice_property_exposes_callable():
    choice = lambda seq: seq[0]
    c = RRCache(2, choice=choice)
    assert c.choice is choice


def test_rr_cache_popitem_uses_custom_choice():
    c = RRCache(3, choice=lambda seq: seq[-1])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert c.popitem() == ("c", 3)


def test_rr_cache_clear_empties_cache():
    c = RRCache(2, choice=lambda seq: seq[0])
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()


def test_keys_hashkey_basic_and_kwargs_order():
    assert keys.hashkey(1, 2) == keys.hashkey(1, 2)
    assert keys.hashkey(a=1, b=2) == keys.hashkey(b=2, a=1)
    assert keys.hashkey(1, a=2) != keys.hashkey(1, a=3)


def test_keys_typedkey_distinguishes_types():
    assert keys.typedkey(1) != keys.typedkey(1.0)
    assert keys.typedkey(a=1) != keys.typedkey(a=1.0)


def test_keys_methodkey_ignores_self_identity():
    class A:
        pass

    a1 = A()
    a2 = A()
    assert keys.methodkey(a1, 1, x=2) == keys.methodkey(a2, 1, x=2)


def test_keys_typedmethodkey_behaves_like_typedkey_for_args():
    class A:
        pass

    a = A()
    assert keys.typedmethodkey(a, 1) == keys.typedkey(1)


def test_cached_without_cache_returns_direct_results_and_cache_clear():
    calls = {"n": 0}

    @cached(None)
    def f(x):
        calls["n"] += 1
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert calls["n"] == 2
    assert f.cache_clear() is None
    assert f.cache_info is None


def test_cached_with_cache_caches_results_and_info():
    c = Cache(10)
    calls = {"n": 0}

    @cached(c, info=True)
    def f(x):
        calls["n"] += 1
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    info = f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize == 10
    assert info.currsize == 1
    assert calls["n"] == 1


def test_cached_with_custom_key_function():
    c = Cache(10)

    @cached(c, key=lambda x, y: (x + y,))
    def f(x, y):
        return x * 100 + y

    assert f(1, 2) == 102
    assert f(2, 1) == 102
    assert len(c) == 1


def test_cached_lock_and_condition_attributes():
    c = Cache(10)
    lock = threading.Lock()
    cond = threading.Condition(lock)

    @cached(c, lock=lock, condition=cond)
    def f(x):
        return x

    assert f.cache is c
    assert f.cache_lock is lock
    assert f.cache_condition is cond


def test_cachedmethod_on_instance_uses_per_instance_cache():
    class A:
        def __init__(self):
            self.cache = Cache(10)
            self.calls = 0

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            self.calls += 1
            return x + 1

    a = A()
    assert a.f(1) == 2
    assert a.f(1) == 2
    assert a.calls == 1
    assert len(a.cache) == 1


def test_cachedmethod_cache_info_for_mapping_cache():
    class A:
        def __init__(self):
            self.cache = {}
            self.calls = 0

        @cachedmethod(lambda self: self.cache, info=True)
        def f(self, x):
            self.calls += 1
            return x * 2

    a = A()
    assert a.f(2) == 4
    assert a.f(2) == 4
    info = a.f.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.maxsize is None
    assert info.currsize == 1


def test_cachedmethod_class_access_returns_wrapper_like_descriptor():
    class A:
        cache = {}

        @cachedmethod(lambda self: self.cache)
        def f(self, x):
            return x

    assert hasattr(A.f, "cache_clear")
    assert A.f.cache is A.cache


def test_cachedmethod_deprecated_classmethod_warns():
    class A:
        cache = {}

        @classmethod
        @cachedmethod(lambda cls: cls.cache)
        def f(cls, x):
            return x + 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert A.f(1) == 2
    assert any(issubclass(item.category, DeprecationWarning) for item in w)


def test_cachedmethod_no_dict_on_slots_class_raises_typeerror():
    class A:
        __slots__ = ()

        def __init__(self):
            pass

        @cachedmethod(lambda self: {})
        def f(self, x):
            return x

    a = A()
    with pytest.raises(TypeError):
        _ = a.f


def test_cachedmethod_set_name_rejects_two_names():
    cm = cachedmethod(lambda self: {})

    class A:
        x = cm
        with pytest.raises(TypeError):
            y = cm


def test_fifo_func_decorator_basic_and_parameters():
    @fifo_cache(2)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert f.cache_parameters() == {"maxsize": 2, "typed": False}
    assert f.cache_info().hits == 1


def test_lfu_func_decorator_basic():
    @lfu_cache(2)
    def f(x):
        return x * 2

    assert f(2) == 4
    assert f(2) == 4
    assert f.cache_info().hits == 1


def test_lru_func_decorator_basic():
    @lru_cache(2)
    def f(x):
        return x * 3

    assert f(2) == 6
    assert f(2) == 6
    assert f.cache_info().hits == 1


def test_rr_func_decorator_with_deterministic_choice():
    @rr_cache(2, choice=lambda seq: seq[0])
    def f(x):
        return x * 4

    assert f(2) == 8
    assert f(2) == 8
    assert f.cache_info().hits == 1


def test_ttl_func_decorator_expires_with_freezegun():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        @ttl_cache(2, ttl=10, timer=lambda: frozen().timestamp())
        def f(x):
            return x + 1

        assert f(1) == 2
        assert f(1) == 2
        frozen.tick(11)
        assert f(1) == 2
        assert f.cache_info().misses == 2


def test_ttlcache_contains_and_getitem_respects_expiration():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TTLCache(10, ttl=5, timer=t)
        c["a"] = 1
        assert "a" in c
        frozen.tick(6)
        assert "a" not in c
        with pytest.raises(KeyError):
            _ = c["a"]


def test_ttlcache_expire_removes_and_returns_items():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TTLCache(10, ttl=5, timer=t)
        c["a"] = 1
        c["b"] = 2
        frozen.tick(6)
        expired = c.expire()
        assert set(expired) == {("a", 1), ("b", 2)}
        assert len(c) == 0


def test_ttlcache_iteration_skips_expired_items():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TTLCache(10, ttl=5, timer=t)
        c["a"] = 1
        frozen.tick(3)
        c["b"] = 2
        frozen.tick(3)
        assert list(c) == ["b"]


def test_ttlcache_popitem_removes_oldest_live_item():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TTLCache(10, ttl=100, timer=t)
        c["a"] = 1
        frozen.tick(1)
        c["b"] = 2
        assert c.popitem() == ("a", 1)


def test_ttlcache_clear_empties_cache():
    with freeze_time("2020-01-01 00:00:00"):
        c = TTLCache(10, ttl=5)
        c["a"] = 1
        c.clear()
        assert len(c) == 0


def test_ttlcache_setstate_rebuilds_order_and_expires():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TTLCache(10, ttl=5, timer=t)
        c["a"] = 1
        c["b"] = 2
        state = c.__dict__.copy()
        new = TTLCache(10, ttl=5, timer=t)
        new.__setstate__(state)
        assert list(new) == ["a", "b"]
        frozen.tick(6)
        assert len(new) == 0


def test_tlru_cache_custom_ttu_and_expire():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()

        def ttu(key, value, now):
            return now + value

        c = TLRUCache(10, ttu=ttu, timer=t)
        c["a"] = 5  # expires at now+5
        c["b"] = 20
        frozen.tick(6)
        assert "a" not in c
        assert "b" in c
        assert c.expire() == [("a", 5)]


def test_tlru_cache_getitem_and_iteration():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TLRUCache(10, ttu=lambda k, v, now: now + 5, timer=t)
        c["a"] = 1
        c["b"] = 2
        assert c["a"] == 1
        frozen.tick(6)
        with pytest.raises(KeyError):
            _ = c["a"]
        assert list(c) == []


def test_tlru_cache_popitem_and_clear():
    with freeze_time("2020-01-01 00:00:00") as frozen:
        t = lambda: frozen().timestamp()
        c = TLRUCache(10, ttu=lambda k, v, now: now + 100, timer=t)
        c["a"] = 1
        c["b"] = 2
        assert c.popitem() == ("a", 1)
        c.clear()
        with pytest.raises(KeyError):
            c.popitem()


def test_tlru_cache_inserting_expired_value_removes_existing():
    with freeze_time("2020-01-01 00:00:00"):
        t = lambda: 0.0
        c = TLRUCache(10, ttu=lambda k, v, now: now + 1, timer=t)
        c["a"] = 1
        c.__setitem__("a", 2)
        assert c["a"] == 2
