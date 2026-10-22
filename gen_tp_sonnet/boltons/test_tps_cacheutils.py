import pytest

from boltons.cacheutils import (
    LRI,
    LRU,
    MinIDMap,
    ThresholdCounter,
    cached,
    cachedmethod,
    cachedproperty,
    make_cache_key,
)


def test_lri_evicts_least_recently_inserted_regardless_of_reads():
    cache = LRI(max_size=2)
    cache['a'], cache['b'] = 'A', 'B'
    for _ in range(3):
        assert cache['a'] == 'A'
    cache['c'] = 'C'
    assert 'a' not in cache
    assert cache.get('a') is None
    assert cache['b'] == 'B'
    assert cache['c'] == 'C'


def test_lru_evicts_least_recently_used_and_respects_max_size():
    cache = LRU(max_size=2)
    cache['a'], cache['b'] = 'A', 'B'
    assert cache['a'] == 'A'
    cache['c'] = 'C'
    assert 'b' not in cache
    assert cache['a'] == 'A'
    assert cache['c'] == 'C'

    big = LRU(max_size=3)
    for i in range(10):
        big[i] = i
    assert len(big) == 3
    assert sorted(big) == [7, 8, 9]


def test_lru_on_miss_and_initial_values():
    calls = []

    def on_miss(key):
        calls.append(key)
        return key * 2

    cache = LRU(max_size=3, values={'seed': 1}, on_miss=on_miss)
    assert cache['seed'] == 1
    assert cache[4] == 8
    assert cache[4] == 8
    assert calls == [4]


def test_lru_hit_and_miss_counts():
    cache = LRU(max_size=2)
    cache['a'] = 1
    assert cache['a'] == 1
    assert cache['a'] == 1
    assert cache.hit_count == 2
    assert cache.get('missing') is None
    assert cache.miss_count == 1


def test_make_cache_key_is_deterministic_and_sensitive_to_arguments():
    k1 = make_cache_key(('a', 'b'), {'c': 'd'})
    k2 = make_cache_key(('a', 'b'), {'c': 'd'})
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert k1 != make_cache_key(('a', 'b'), {'c': 'e'})
    assert k1 != make_cache_key(('a', 'x'), {'c': 'd'})
    assert make_cache_key((), {'x': 1, 'y': 2}) == make_cache_key((), {'y': 2, 'x': 1})


def test_make_cache_key_single_fast_type_argument_is_returned_bare():
    assert make_cache_key((3,), {}) == 3
    assert make_cache_key(('abc',), {}) == 'abc'
    assert make_cache_key((None,), {}) is None


def test_make_cache_key_typed_distinguishes_int_and_float():
    seen = {make_cache_key((3,), {}, typed=True): 'int'}
    assert make_cache_key((3.0,), {}, typed=True) not in seen
    assert seen[make_cache_key((3,), {}, typed=True)] == 'int'


def test_cached_function_only_calls_underlying_once_per_argument():
    calls = []

    @cached({})
    def double(x):
        calls.append(x)
        return x * 2

    assert double(2) == 4
    assert double(2) == 4
    assert double(3) == 6
    assert calls == [2, 3]


def test_cached_with_lru_cache_object():
    calls = []
    cache = LRU(max_size=8)

    @cached(cache)
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert add(1) == 1
    assert calls == [(1, 2), (1, 0)]
    assert len(cache) == 2


def test_cachedmethod_with_attribute_name_is_per_instance():
    class Squarer:
        def __init__(self):
            self._cache = {}
            self.calls = 0

        @cachedmethod('_cache')
        def square(self, x):
            self.calls += 1
            return x * x

    a, b = Squarer(), Squarer()
    assert a.square(3) == 9
    assert a.square(3) == 9
    assert a.calls == 1
    assert b.square(3) == 9
    assert b.calls == 1


def test_cachedproperty_computes_once_and_can_be_cleared_with_delattr():
    class Thing:
        def __init__(self):
            self.count = 0

        @cachedproperty
        def value(self):
            self.count += 1
            return 42

    t = Thing()
    assert t.value == 42
    assert t.value == 42
    assert t.count == 1
    delattr(t, 'value')
    assert t.value == 42
    assert t.count == 2


def test_threshold_counter_documented_behavior():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_threshold_counter_most_common_and_totals():
    tc = ThresholdCounter()
    tc.update(['a', 'a', 'b', 'b', 'b', 'c'])
    assert tc.most_common(2) == [('b', 3), ('a', 2)]
    assert tc.most_common()[0] == ('b', 3)
    assert tc.get('zzz') == 0
    assert tc.get_common_count() == 6
    assert tc.get_uncommon_count() == 0
    assert 0 < tc.get_commonality() <= 1.0


def test_min_id_map_assigns_smallest_free_ids():
    class Obj:
        pass

    ids = MinIDMap()
    a, b, c = Obj(), Obj(), Obj()
    assert [ids.get(a), ids.get(b), ids.get(c)] == [0, 1, 2]
    assert ids.get(a) == 0
    ids.drop(b)
    d = Obj()
    assert ids.get(d) == 1
