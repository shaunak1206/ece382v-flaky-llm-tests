import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    ThresholdCounter,
    cached,
    cachedmethod,
    cachedproperty,
    make_cache_key,
)


def test_lri_evicts_oldest_insert():
    cache = LRI(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    assert len(cache) == 2
    assert 'a' not in cache
    assert dict(cache) == {'b': 2, 'c': 3}


def test_lri_read_does_not_refresh():
    cache = LRI(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1
    cache['c'] = 3
    # 'a' was inserted first, so it is evicted despite the recent read
    assert 'a' not in cache
    assert set(cache) == {'b', 'c'}


def test_lru_read_refreshes():
    cache = LRU(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1
    cache['c'] = 3
    assert 'b' not in cache
    assert set(cache) == {'a', 'c'}


def test_cache_stats_and_on_miss():
    cache = LRU(max_size=2)
    cache['a'], cache['b'] = 'A', 'B'
    for _ in range(3):
        assert cache['b'] == 'B'
    cache['c'] = 'C'
    assert cache.get('a') is None
    assert (cache.hit_count, cache.miss_count, cache.soft_miss_count) == (3, 1, 1)

    filled = LRU(on_miss=lambda k: k * 2)
    assert filled[3] == 6
    assert 3 in filled
    assert filled[3] == 6
    assert filled.miss_count == 1
    assert filled.hit_count == 1


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        LRI(max_size=0)
    with pytest.raises(ValueError):
        LRU(max_size=-5)
    with pytest.raises(TypeError):
        LRI(on_miss='not callable')


def test_setdefault_pop_and_popitem():
    cache = LRI(max_size=2)
    assert cache.setdefault('a', 5) == 5
    assert cache.setdefault('a', 99) == 5
    assert cache['a'] == 5

    cache['b'] = 6
    assert cache.pop('a') == 5
    assert cache.pop('missing', 'dflt') == 'dflt'
    with pytest.raises(KeyError):
        cache.pop('missing')

    cache['c'] = 7
    cache['d'] = 8
    assert dict(cache) == {'c': 7, 'd': 8}

    key, value = cache.popitem()
    assert key not in cache
    assert len(cache) == 1


def test_clear_copy_and_update():
    cache = LRU(max_size=3, values={'a': 1})
    cache.update({'b': 2}, c=3)
    assert dict(cache) == {'a': 1, 'b': 2, 'c': 3}

    dup = cache.copy()
    assert dup == cache
    assert dup is not cache
    assert dup.max_size == 3
    dup['z'] = 26
    assert 'z' in dup
    assert len(dup) == 3
    assert 'a' not in dup
    assert 'z' not in cache

    cache.clear()
    assert len(cache) == 0
    cache['x'] = 1
    assert dict(cache) == {'x': 1}


def test_make_cache_key():
    assert make_cache_key((1,), {}) == 1
    assert make_cache_key(('a',), {}) == 'a'
    multi = make_cache_key((1, 2), {})
    assert list(multi)[:2] == [1, 2]
    assert hash(multi) == hash((1, 2))
    assert make_cache_key((1, 2), {}) == multi

    with_kw = make_cache_key(('a',), {'b': 1, 'a': 2})
    assert ('a', 2) in with_kw
    assert ('b', 1) in with_kw
    assert with_kw == make_cache_key(('a',), {'a': 2, 'b': 1})

    assert (make_cache_key((3,), {}, typed=True)
            != make_cache_key((3.0,), {}, typed=True))


def test_cached_decorator():
    cache = LRU()
    calls = []

    @cached(cache)
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert square(4) == 16
    assert calls == [3, 4]
    assert len(cache) == 2


def test_cachedmethod_with_attribute_cache():
    class Lowerer:
        def __init__(self):
            self.cache = LRI()
            self.calls = 0

        @cachedmethod('cache')
        def lower(self, text):
            self.calls += 1
            return text.lower()

    obj = Lowerer()
    assert obj.lower('ABC') == 'abc'
    assert obj.lower('ABC') == 'abc'
    assert obj.calls == 1
    assert len(obj.cache) == 1
    assert obj.lower('DEF') == 'def'
    assert obj.calls == 2


def test_cachedproperty_computes_once():
    class Thing:
        def __init__(self):
            self.calls = 0

        @cachedproperty
        def value(self):
            self.calls += 1
            return 42

    thing = Thing()
    assert thing.value == 42
    assert thing.value == 42
    assert thing.calls == 1
    assert 'value' in thing.__dict__
    delattr(thing, 'value')
    assert thing.value == 42
    assert thing.calls == 2


def test_threshold_counter_basic():
    tc = ThresholdCounter(threshold=0.1)
    tc.update([1, 1, 2])
    assert tc.total == 3
    assert tc[1] == 2
    assert tc.get(2) == 1
    assert tc.get(99) == 0
    assert tc.get(99, -1) == -1
    assert 1 in tc
    assert len(tc) == 2
    assert tc.most_common(1) == [(1, 2)]
    assert sorted(tc.elements()) == [1, 1, 2]
    assert tc.threshold == 0.1
    assert tc.get_common_count() == 3
    assert tc.get_uncommon_count() == 0

    tc.update({'x': 2})
    assert tc['x'] == 2

    for bad in (0, 1, -0.5, 2):
        with pytest.raises(ValueError):
            ThresholdCounter(threshold=bad)


def test_threshold_counter_compaction():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    tc.update([2] * 10)
    # compaction after the 10th addition culls the rare key 1
    assert tc.get(1) == 0
    assert 1 not in tc
    assert tc.get(2) == 10
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11
    assert tc.total == 12
