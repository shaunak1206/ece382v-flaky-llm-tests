import collections
import threading
import time
import random
import pytest
import math
from types import SimpleNamespace
from freezegun import freeze_time

import cachetools
from cachetools import (
    Cache, FIFOCache, LFUCache, LRUCache, RRCache, TTLCache, TLRUCache,
    cached, cachedmethod,
)
from cachetools._cached import _wrapper as func_wrapper
from cachetools._cachedmethod import _wrapper as method_wrapper
from cachetools import keys


# Helper function for getsizeof counting by length of string/int
def getsizeof_len(value):
    return len(str(value))


# ====================
# Tests for Cache base
# ====================

def test_cache_init_invalid_maxsize():
    with pytest.raises(ValueError):
        Cache(-1)


def test_cache_set_and_get_item_with_default_getsizeof():
    cache = Cache(3)
    cache['a'] = 1
    assert cache['a'] == 1
    assert len(cache) == 1
    assert 1 == cache.currsize


def test_cache_setitem_pop_reduces_size_appropriately():
    cache = Cache(10, getsizeof_len)
    cache['a'] = "123"
    cache['b'] = "12"
    assert cache.currsize == 5
    val = cache.pop('a')
    assert val == "123"
    assert 'a' not in cache
    assert cache.currsize == 2


def test_cache_setitem_raises_valueerror_for_negative_size():
    cache = Cache(10, lambda v: -1)
    with pytest.raises(ValueError):
        cache['a'] = 'x'


def test_cache_setitem_raises_valueerror_when_value_too_large():
    cache = Cache(3, getsizeof_len)
    with pytest.raises(ValueError):
        cache['a'] = 'abcd'


def test_cache_eviction_on_setitem_exceeding_maxsize():
    cache = Cache(3)
    cache['a'] = 1
    cache['b'] = 1
    cache['c'] = 1
    # Should evict items via popitem until size fits
    # But base Cache does not implement popitem - so expect AttributeError on eviction
    # So patch popitem temporarily

    def popitem_stub():
        # evict 'a'
        key = next(iter(cache))
        cache.__delitem__(key)
        return key, 1

    cache.popitem = popitem_stub
    cache['d'] = 2
    assert 'd' in cache
    assert len(cache) <= 3


def test_cache_delitem_and_contains():
    cache = Cache(10)
    cache['x'] = 42
    assert 'x' in cache
    del cache['x']
    assert 'x' not in cache
    with pytest.raises(KeyError):
        _ = cache['x']


def test_cache_missing_raises_keyerror():
    cache = Cache(10)
    with pytest.raises(KeyError):
        cache.__missing__('missing_key')


def test_cache_get_and_setdefault():
    cache = Cache(5)
    assert cache.get('missing') is None
    assert cache.get('missing', 123) == 123

    cache.setdefault('a', 5)
    assert cache['a'] == 5

    # setdefault returns existing value
    result = cache.setdefault('a', 10)
    assert result == 5
    assert cache['a'] == 5


def test_cache_pop_with_default():
    cache = Cache(5)
    cache['a'] = 1
    popped = cache.pop('a')
    assert popped == 1
    # pop again with default
    popped2 = cache.pop('a', 42)
    assert popped2 == 42
    with pytest.raises(KeyError):
        cache.pop('b')


def test_cache_clear_empties_cache():
    cache = Cache(5)
    cache['a'] = 1
    cache['b'] = 2
    cache.clear()
    assert len(cache) == 0
    assert cache.currsize == 0


def test_cache_repr_and_properties():
    cache = Cache(5)
    cache['a'] = 1
    r = repr(cache)
    assert "Cache" in r
    assert cache.maxsize == 5
    assert cache.currsize == 1


# ===================
# Tests for FIFOCache
# ===================

def test_fifo_cache_set_and_getitem():
    fifo = FIFOCache(3)
    fifo['a'] = 1
    fifo['b'] = 2
    fifo['c'] = 3
    assert fifo['a'] == 1
    # order keys preserved
    keys = list(fifo._FIFOCache__order.keys())
    assert keys == ['a', 'b', 'c']


def test_fifo_cache_popitem_and_clear():
    fifo = FIFOCache(3)
    fifo['a'] = 1
    fifo['b'] = 2
    key, value = fifo.popitem()
    assert (key, value) in [('a', 1), ('b', 2)]
    fifo.clear()
    assert len(fifo) == 0
    assert len(fifo._FIFOCache__order) == 0


def test_fifo_cache_delitem_removes_order():
    fifo = FIFOCache(3)
    fifo['a'] = 1
    del fifo['a']
    assert 'a' not in fifo
    assert 'a' not in fifo._FIFOCache__order


def test_fifo_cache_raises_keyerror_on_empty_popitem():
    fifo = FIFOCache(1)
    with pytest.raises(KeyError):
        fifo.popitem()


# ===================
# Tests for LFUCache
# ===================

def test_lfu_cache_set_and_getitem_updates_links():
    lfu = LFUCache(3)
    lfu['a'] = 1
    lfu['b'] = 2
    val_a = lfu['a']
    # After access, a's link count should increase
    links = lfu._LFUCache__links
    link_a = links['a']
    assert link_a.count >= 1
    assert val_a == 1


def test_lfu_cache_popitem_removes_least_frequent():
    lfu = LFUCache(3)
    lfu['a'] = 1
    lfu['b'] = 2
    lfu['c'] = 3
    lfu['a']  # increases frequency for 'a'
    key, val = lfu.popitem()
    # The least frequently used (other than 'a') should be evicted ('b' or 'c')
    assert key in ('b', 'c')
    assert val in (2, 3)


def test_lfu_cache_delitem_removes_links_and_keys():
    lfu = LFUCache(3)
    lfu['a'] = 1
    del lfu['a']
    assert 'a' not in lfu
    assert 'a' not in lfu._LFUCache__links


def test_lfu_cache_clear_resets_links():
    lfu = LFUCache(3)
    lfu['a'] = 1
    lfu.clear()
    assert len(lfu) == 0
    assert not lfu._LFUCache__links


def test_lfu_cache_touch_increments_count_and_moves_keys():
    lfu = LFUCache(3)
    lfu['a'] = 1
    old_link = lfu._LFUCache__links['a']
    old_count = old_link.count
    lfu._LFUCache__touch('a')
    new_link = lfu._LFUCache__links['a']
    assert new_link.count == old_count + 1 or (old_count == 0 and new_link.count == 1)


# ===================
# Tests for LRUCache
# ===================

def test_lru_cache_set_get_touch_order():
    lru = LRUCache(3)
    lru['a'] = 1
    lru['b'] = 2
    lru['c'] = 3

    # Access a to move it to the end (most recent)
    _ = lru['a']
    order = list(lru._LRUCache__order.keys())
    # 'a' should be last
    assert order[-1] == 'a'


def test_lru_cache_popitem_removes_least_recent():
    lru = LRUCache(3)
    lru['a'] = 1
    lru['b'] = 2
    lru['c'] = 3
    lru['a']  # access a
    key, val = lru.popitem()
    # least recently used is b or c (a was moved recently)
    assert key in ('b', 'c')


def test_lru_cache_delitem_removes_order():
    lru = LRUCache(3)
    lru['a'] = 1
    del lru['a']
    assert 'a' not in lru._LRUCache__order


def test_lru_cache_clear_resets_order():
    lru = LRUCache(3)
    lru['a'] = 1
    lru.clear()
    assert not bool(lru._LRUCache__order)
    assert len(lru) == 0


# ===================
# Tests for RRCache
# ===================

def test_rr_cache_inserts_and_maintains_index():
    chosen = None
    def choice(seq):
        nonlocal chosen
        chosen = seq[0]
        return seq[0]
    rr = RRCache(5, choice)
    rr['a'] = 1
    rr['b'] = 2
    assert rr['a'] == 1
    assert rr['b'] == 2
    assert rr._RRCache__index['a'] == 0
    assert rr._RRCache__index['b'] == 1
    assert rr._RRCache__keys == ['a', 'b']


def test_rr_cache_popitem_removes_random_key(monkeypatch):
    rr = RRCache(5, choice=lambda seq: seq[0])
    rr['a'] = 1
    rr['b'] = 2
    key, val = rr.popitem()
    assert key == 'a'
    assert val == 1
    assert 'a' not in rr


def test_rr_cache_delitem_removes_key_and_updates_indices():
    rr = RRCache(5)
    rr['a'] = 1
    rr['b'] = 2
    rr['c'] = 3
    idx_b = rr._RRCache__index['b']
    del rr['b']
    # Keys length reduce by one
    assert len(rr._RRCache__keys) == 2
    assert 'b' not in rr._RRCache__index
    # Indices update accordingly
    for i, key in enumerate(rr._RRCache__keys):
        assert rr._RRCache__index[key] == i


def test_rr_cache_clear_resets_all():
    rr = RRCache(5)
    rr['a'] = 1
    rr.clear()
    assert len(rr) == 0
    assert not rr._RRCache__index
    assert not rr._RRCache__keys


def test_rr_cache_popitem_raises_keyerror_when_empty():
    rr = RRCache(5)
    with pytest.raises(KeyError):
        rr.popitem()


# =================
# Tests for _TimedCache
# =================

def test_timed_cache_expire_not_implemented():
    class TestTimed(_TimedCache):
        def expire(self, time=None):
            return super().expire(time)
    t = TestTimed(5, timer=time.monotonic)
    with pytest.raises(NotImplementedError):
        t.expire()


def test_timed_cache_timer_property_and_currsize_calls_expire(mocker):
    called = {}
    def fake_timer():
        called['timer'] = True
        return 100
    class SubTimed(_TimedCache):
        def expire(self, time=None):
            called['expire'] = time
            return []

    c = SubTimed(5, timer=fake_timer)
    _ = c.timer()
    _ = c.currsize
    assert called.get('timer') == True
    assert called.get('expire') == 100


def test_timed_cache_get_pop_setdefault_use_timer():
    class SubTimed(_TimedCache):
        def expire(self, time=None):
            return []

    c = SubTimed(5, timer=time.monotonic)
    c['a'] = 'a'
    assert c.get('a') == 'a'
    assert c.setdefault('b', 'b') == 'b'
    assert c.pop('a') == 'a'


# =========
# Tests for TTLCache
# =========

def test_ttlcache_set_and_getitem_and_contains():
    ttl = TTLCache(10, ttl=1)
    ttl['a'] = 1
    assert 'a' in ttl
    assert ttl['a'] == 1
    # Sleep to expire the entry
    time.sleep(1.1)
    assert 'a' not in ttl
    with pytest.raises(KeyError):
        ttl['a']


def test_ttlcache_delitem_raises_when_expired():
    ttl = TTLCache(10, ttl=0.5)
    ttl['a'] = 1
    del ttl['a']
    with pytest.raises(KeyError):
        # After deletion, re-accessing raises KeyError
        _ = ttl['a']


def test_ttlcache_expire_removes_expired_entries():
    ttl = TTLCache(10, ttl=0.1)
    ttl['a'] = 1
    time.sleep(0.2)
    expired = ttl.expire()
    assert expired[0][0] == 'a'
    assert 'a' not in ttl


def test_ttlcache_iterator_yields_unexpired_only():
    ttl = TTLCache(10, ttl=0.5)
    ttl['a'] = 1
    ttl['b'] = 2
    time.sleep(0.6)
    l = list(ttl)
    assert l == []


def test_ttlcache_popitem_removes_least_recent_valid_item():
    ttl = TTLCache(10, ttl=0.1)
    ttl['a'] = 1
    time.sleep(0.2)
    with pytest.raises(KeyError):
        ttl.popitem()

    ttl['b'] = 2
    key, val = ttl.popitem()
    assert (key, val) == ('b', 2)


def test_ttlcache_clear_and_getlink_behavior():
    ttl = TTLCache(10, ttl=60)
    ttl['abc'] = 123
    assert isinstance(ttl._TTLCache__links['abc'], TTLCache._Link)
    ttl.clear()
    assert len(ttl) == 0


# =========
# Tests for TLRUCache
# =========

def test_tlru_cache_set_get_contains_and_expire():
    calls = []

    def timer():
        calls.append('timer')
        return time.monotonic()

    def ttu(key, value, now):
        return now + 0.1

    tlru = TLRUCache(10, ttu, timer=timer)
    tlru['a'] = 10
    assert 'a' in tlru
    val = tlru['a']
    assert val == 10
    time.sleep(0.15)
    expired = tlru.expire()
    assert ('a', 10) in expired
    assert 'a' not in tlru


def test_tlru_cache_popitem_behaves_correctly():
    def ttu(key, val, now):
        return now + 1
    tlru = TLRUCache(5, ttu)
    tlru['x'] = 5
    key, val = tlru.popitem()
    assert (key, val) == ('x', 5)
    with pytest.raises(KeyError):
        tlru.popitem()


def test_tlru_cache_clear_empties_all():
    def ttu(key, val, now):
        return now + 1
    tlru = TLRUCache(5, ttu)
    tlru['x'] = 5
    tlru.clear()
    assert len(tlru) == 0


def test_tlru_cache_iter_yields_only_non_expired():
    def ttu(key, val, now):
        return now + 0.1
    tlru = TLRUCache(5, ttu)
    tlru['x'] = 1
    tlru['y'] = 2
    time.sleep(0.15)
    result = list(tlru)
    assert result == []


# ===========================
# Tests for cached function decorator
# ===========================

def test_cached_decorator_basic_usage():
    calls = 0

    @cached(Cache(2))
    def f(x):
        nonlocal calls
        calls += 1
        return x * 2

    assert f(1) == 2
    assert f(1) == 2
    assert calls == 1
    assert f(2) == 4
    assert calls == 2
    assert f(2) == 4
    assert calls == 2


def test_cached_decorator_with_info():
    from cachetools import _CacheInfo

    cache = Cache(2)

    @cached(cache, info=True)
    def g(x):
        return x + 1

    g(1)
    g(2)
    g(1)
    info = g.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 2
    assert info.currsize == 2


# =========================
# Tests for cachedmethod decorator
# =========================

class Dummy:
    def __init__(self):
        self._cache = Cache(2)

    @cachedmethod(lambda self: self._cache)
    def times_two(self, x):
        return x * 2


def test_cachedmethod_basic():
    d = Dummy()
    assert d.times_two(3) == 6
    assert d.times_two(3) == 6


def test_cachedmethod_info():
    class DummyInfo:
        def __init__(self):
            self._cache = Cache(2)

        @cachedmethod(lambda self: self._cache, info=True)
        def add(self, x):
            return x + 1

    d = DummyInfo()
    d.add(1)
    d.add(2)
    d.add(3)
    d.add(2)
    info = d.add.cache_info()
    assert info.hits == 1
    assert info.misses == 3


# =========================
# Tests for keys module functions
# =========================


def test_hashkey_and_methodkey_equivalence():
    k1 = keys.hashkey(1, 2, a=3)
    k2 = keys.methodkey(None, 1, 2, a=3)
    assert k1 == k2
    assert isinstance(k1, tuple)


def test_typedkey_includes_types():
    k = keys.typedkey(1, 2, a=3)
    diffs = tuple(type(x) for x in (1, 2)) + (int,)
    assert all(isinstance(element, type) for element in k[-len(diffs):])


def test_typedmethodkey_same_as_typedkey():
    k1 = keys.typedmethodkey(None, 1, 2, a=3)
    k2 = keys.typedkey(1, 2, a=3)
    assert k1 == k2


# =========================
# Tests for func.py caching decorators
# =========================

def test_fifo_cache_decorator_with_defaults():
    calls = 0

    @cachetools.func.fifo_cache(maxsize=3)
    def f(x):
        nonlocal calls
        calls += 1
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert calls == 1


def test_lfu_cache_decorator_with_defaults():
    calls = 0

    @cachetools.func.lfu_cache(maxsize=3)
    def f(x):
        nonlocal calls
        calls += 1
        return x

    assert f(1) == 1
    assert f(2) == 2
    assert f(1) == 1
    assert calls == 2


def test_lru_cache_decorator_with_defaults():
    calls = 0

    @cachetools.func.lru_cache(maxsize=3)
    def f(x):
        nonlocal calls
        calls += 1
        return x

    assert f(3) == 3
    assert f(3) == 3
    assert calls == 1


def test_rr_cache_decorator_with_defaults():
    calls = 0

    @cachetools.func.rr_cache(maxsize=3, choice=lambda x: x[0])
    def f(x):
        nonlocal calls
        calls += 1
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert calls == 1


def test_ttl_cache_decorator_with_defaults():
    calls = 0

    @cachetools.func.ttl_cache(maxsize=3, ttl=2)
    def f(x):
        nonlocal calls
        calls += 1
        return x + 2

    assert f(1) == 3
    assert f(1) == 3
    assert calls == 1
    time.sleep(2.1)
    assert f(1) == 3
    assert calls == 2


# =========================
# Tests for _cached.py functional wrappers
# =========================

def test_uncached_function_wrapper_counts_misses():
    from cachetools._cached import _uncached_info

    misses = 0
    def info(hits, misses_):
        nonlocal misses
        misses = misses_
        return (hits, misses_)

    calls = 0
    def f(x):
        nonlocal calls
        calls += 1
        return x

    wrapper = _uncached_info(f, info)
    assert wrapper(1) == 1
    assert wrapper.cache_info() == (0, 1)
    wrapper.cache_clear()
    assert misses == 0


def test_unlocked_info_records_hits_and_misses():
    from cachetools._cached import _unlocked_info
    c = Cache(10)
    def info(h, m):
        return (h, m)

    def f(x):
        return x

    wrapper = _unlocked_info(f, c, keys.hashkey, info)
    assert wrapper(0) == 0
    assert wrapper(0) == 0  # cache hit
    h, m = wrapper.cache_info()
    assert h == 1
    assert m == 1


def test_locked_info_records_hits_and_misses():
    from cachetools._cached import _locked_info
    lock = threading.Lock()
    c = Cache(10)
    def info(h, m):
        return (h, m)

    def f(x):
        return x

    wrapper = _locked_info(f, c, keys.hashkey, lock, info)
    assert wrapper(1) == 1
    assert wrapper(1) == 1
    h, m = wrapper.cache_info()
    assert h == 1
    assert m == 1


def test_condition_info_wrapper_behavior():
    from cachetools._cached import _condition_info
    lock = threading.Lock()
    cond = threading.Condition(lock)
    c = Cache(10)
    def info(h, m):
        return (h, m)

    def f(x):
        return x

    wrapper = _condition_info(f, c, keys.hashkey, lock, cond, info)
    assert wrapper(2) == 2
    assert wrapper(2) == 2
    h, m = wrapper.cache_info()
    assert h == 1
    assert m == 1


# =========================
# Tests for _cachedmethod.py basics
# =========================

def test_uncached_method_wrapper_basic():
    from cachetools._cachedmethod import _uncached_info
    hits = 0
    misses = 0
    def info(cache, h, m):
        nonlocal hits, misses
        hits = h
        misses = m
        return h, m

    class A:
        @cachedmethod(cache=lambda self: Cache(5), info=True)
        def f(self, x):
            return x

    a = A()
    assert a.f(1) == 1
    assert a.f(1) == 1
    info = a.f.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_cachedmethod_raises_deprecated_warning_for_classmethod():
    import warnings
    from cachetools._cachedmethod import _warn_classmethod

    with pytest.warns(DeprecationWarning):
        _warn_classmethod(1)


def test_cachedmethod_descriptor_set_name_and_get(monkeypatch):
    # We test Descriptor __set_name__ and __get__ behavior
    from cachetools._cachedmethod import _DescriptorBase

    class DummyDescriptor(_DescriptorBase):
        def __init__(self):
            super().__init__()
            self.called = False
        def Wrapper(self, obj):
            self.called = True
            return "wrapped"

    desc = DummyDescriptor()
    desc.__set_name__(type(DummyDescriptor), 'name')
    res = desc.__get__(SimpleNamespace(), SimpleNamespace)
    # It should call our Wrapper
    # (res is property dependent, here "wrapped")
    assert res == "wrapped"


def test_cachedmethod_descriptor_set_name_conflict():
    from cachetools._cachedmethod import _DescriptorBase
    desc = _DescriptorBase()
    desc._DescriptorBase__attrname = 'a'
    with pytest.raises(TypeError):
        desc.__set_name__(type(desc), 'b')


@pytest.mark.asyncio
async def test_dummy_async_with_cachedmethod():
    # The library itself doesn't provide async wrappers, so just sanity test sync
    class Dummy:
        def __init__(self):
            self._cache = Cache(5)

        @cachedmethod(lambda self: self._cache)
        def method(self, x):
            return x + 1

    d = Dummy()
    assert d.method(1) == 2


@pytest.mark.parametrize("cache_cls", [FIFOCache, LFUCache, LRUCache, RRCache])
def test_cache_maxsize_and_currsize_properties(cache_cls):
    cache = cache_cls(10)
    cache['a'] = 1
    cache['b'] = 2
    assert cache.maxsize == 10
    assert cache.currsize > 0


def test_cached_decorator_caches_exceptions(monkeypatch):
    # Decorated function raising ValueError should not cache the exception
    calls = 0

    @cached(Cache(5))
    def f(x):
        nonlocal calls
        calls += 1
        if x == 0:
            raise ValueError("Bad")
        return x

    with pytest.raises(ValueError):
        f(0)
    with pytest.raises(ValueError):
        f(0)
    assert calls == 2  # exception not cached


def test_cachedmethod_cache_clear_and_cache_info():
    class D:
        def __init__(self):
            self._cache = Cache(5)

        @cachedmethod(lambda self: self._cache, info=True)
        def m(self, x):
            return x+1

    d = D()
    d.m(1)
    d.m(1)
    info1 = d.m.cache_info()
    d.m.cache_clear()
    info2 = d.m.cache_info()
    assert info1.hits == 1
    assert info2.hits == 0
    assert info2.misses == 0


def test_ttlcache_delitem_raises_keyerror_when_expired():
    ttl = TTLCache(10, ttl=0.01)
    ttl['a'] = 1
    time.sleep(0.02)
    # Delete expired item triggers KeyError
    with pytest.raises(KeyError):
        del ttl['a']


def test_tlru_cache_delete_raises_keyerror_on_expired():
    def ttu(key, val, now):
        return now + 0.01
    tlru = TLRUCache(5, ttu)
    tlru['x'] = 5
    time.sleep(0.02)
    with pytest.raises(KeyError):
        del tlru['x']


def test_tlru_cache_expire_cleans_up_removed_items():
    def ttu(key, val, now):
        return now + 10
    tlru = TLRUCache(5, ttu)

    # Add items and mark many as removed
    for i in range(10):
        tlru['k' + str(i)] = i
    for key in list(tlru._TLRUCache__items.keys())[:7]:
        tlru._TLRUCache__items[key].removed = True

    before_len = len(tlru._TLRUCache__order)
    tlru.expire()
    after_len = len(tlru._TLRUCache__order)
    # assert heap cleaned up (shortened)
    assert after_len < before_len


def test_fifo_cache_order_moves_to_end_on_update():
    fifo = FIFOCache(3)
    fifo['a'] = 1
    fifo['b'] = 2
    keys_before = list(fifo._FIFOCache__order.keys())
    fifo['a'] = 3  # Update 'a' should move to end
    keys_after = list(fifo._FIFOCache__order.keys())
    assert keys_after[-1] == 'a'
    assert keys_before != keys_after


def test_lru_cache_touch_on_getitem_moves_key_to_end():
    lru = LRUCache(3)
    lru['a'] = 1
    lru['b'] = 2
    order_before = list(lru._LRUCache__order.keys())
    _ = lru['a']
    order_after = list(lru._LRUCache__order.keys())
    assert order_after[-1] == 'a'
    assert order_before != order_after


def test_ttlcache_getitem_with_expired_entry_returns_keyerror():
    ttl = TTLCache(5, ttl=0.1)
    ttl['a'] = 1
    # Force expire by advancing time
    with freeze_time() as frozen:
        frozen.tick(delta=1)
        with pytest.raises(KeyError):
            _ = ttl['a']


def test_ttlcache_expire_returns_expired_entries():
    ttl = TTLCache(5, ttl=0.1)
    with freeze_time() as frozen:
        ttl['a'] = 1
        frozen.tick(delta=0.2)
        expired = ttl.expire()
        assert expired == [('a', 1)]


def test_fifo_cache_popitem_order_first_in_first_out():
    fifo = FIFOCache(3)
    fifo['x'] = 1
    fifo['y'] = 2
    k, v = fifo.popitem()
    assert k == 'x'
    assert v == 1


def test_lfu_cache_counts_usage_and_evicts_least_freq_used():
    lfu = LFUCache(2)
    lfu['x'] = 1
    lfu['y'] = 2
    lfu['x']
    k, v = lfu.popitem()
    assert (k, v) == ('y', 2)


def test_rr_cache_popitem_removes_random_key(monkeypatch):
    rr = RRCache(2, choice=lambda keys: keys[0])
    rr['a'] = 1
    rr['b'] = 2
    k, v = rr.popitem()
    assert k == 'a'


def test_tlru_cache_expire_removes_expired_and_skips_removed():
    def ttu(key, val, now):
        return now + 0.01
    tlru = TLRUCache(5, ttu)
    tlru['a'] = 1
    tlru._TLRUCache__items['a'].removed = True
    time.sleep(0.02)
    expired = tlru.expire()
    assert not expired  # removed item should not appear in expired


def test_cache_setitem_replacement_changes_size():
    cache = Cache(10, getsizeof_len)
    cache['a'] = "123"
    old_size = cache.currsize
    cache['a'] = "12"
    assert cache.currsize < old_size


def test_cache_setitem_with_large_value_removes_items(monkeypatch):
    cache = Cache(3)
    evicted = []

    def popitem_stub():
        key = next(iter(cache))
        evicted.append(key)
        cache.__delitem__(key)
        return key, 1

    cache.popitem = popitem_stub
    cache['a'] = 1
    cache['b'] = 1
    cache['c'] = 1
    # Insert an item larger than one, should pop items first
    cache['d'] = 2
    assert 'd' in cache
    assert evicted


# End of tests
