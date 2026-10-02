"""Unit tests for boltons.cacheutils and boltons.queueutils."""
import pytest

from boltons.cacheutils import (
    LRI, LRU, MinIDMap, ThresholdCounter, cached, cachedmethod,
    cachedproperty, make_cache_key)
from boltons.queueutils import (
    HeapPriorityQueue, PriorityQueue, SortedPriorityQueue)


# ---------------------------------------------------------------- LRU / LRI

def test_lru_evicts_least_recently_used():
    cache = LRU(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1          # touch 'a' so 'b' is now the oldest
    cache['c'] = 3
    assert 'b' not in cache
    assert set(cache) == {'a', 'c'}


def test_lri_evicts_least_recently_inserted():
    cache = LRI(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1          # reads do not refresh LRI entries
    cache['c'] = 3
    assert 'a' not in cache
    assert set(cache) == {'b', 'c'}


@pytest.mark.parametrize('cls', [LRI, LRU])
@pytest.mark.parametrize('bad_size', [0, -1])
def test_cache_rejects_non_positive_max_size(cls, bad_size):
    with pytest.raises(ValueError):
        cls(max_size=bad_size)


@pytest.mark.parametrize('cls', [LRI, LRU])
def test_cache_statistics(cls):
    cache = cls(max_size=4)
    cache['a'] = 1
    assert cache['a'] == 1
    assert cache['a'] == 1
    assert cache.get('missing') is None
    with pytest.raises(KeyError):
        cache['nope']
    assert cache.hit_count == 2
    assert cache.miss_count == 2
    assert cache.soft_miss_count == 1


def test_lru_on_miss_populates_cache():
    calls = []

    def loader(key):
        calls.append(key)
        return key * 2

    cache = LRU(max_size=3, on_miss=loader)
    assert cache[4] == 8
    assert cache[4] == 8
    assert calls == [4]
    assert cache.miss_count == 1
    assert cache.hit_count == 1
    assert len(cache) == 1


def test_cache_on_miss_must_be_callable():
    with pytest.raises(TypeError):
        LRU(on_miss='not callable')


def test_lru_setdefault_pop_and_delete():
    cache = LRU(max_size=5, values={'x': 1})
    assert cache.setdefault('x', 99) == 1
    assert cache.setdefault('y', 2) == 2
    assert cache.pop('x') == 1
    assert cache.pop('x', 'gone') == 'gone'
    with pytest.raises(KeyError):
        cache.pop('x')
    del cache['y']
    assert len(cache) == 0
    with pytest.raises(KeyError):
        del cache['y']


def test_lru_update_copy_and_clear():
    cache = LRU(max_size=3)
    cache.update({'a': 1, 'b': 2})
    cache.update([('c', 3)], d=4)       # 'a' falls off the end
    assert set(cache) == {'b', 'c', 'd'}
    clone = cache.copy()
    assert clone == cache
    assert clone.max_size == cache.max_size
    cache.clear()
    assert len(cache) == 0
    assert len(clone) == 3


def test_lru_popitem_removes_entry():
    cache = LRU(max_size=3, values=[('a', 1)])
    assert cache.popitem() == ('a', 1)
    assert len(cache) == 0
    cache['b'] = 2                       # linked list must still be consistent
    assert cache['b'] == 2


# ------------------------------------------------------------------- cached

def test_cached_memoizes_function_calls():
    calls = []
    cache = LRU()

    @cached(cache)
    def square(x):
        calls.append(x)
        return x * x

    assert square(4) == 16
    assert square(4) == 16
    assert square(5) == 25
    assert calls == [4, 5]
    assert len(cache) == 2


def test_cached_distinguishes_keyword_arguments():
    calls = []

    @cached({})
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert add(1, b=3) == 4
    assert len(calls) == 2


def test_cached_typed_separates_int_and_float():
    calls = []

    @cached(LRU(), typed=True)
    def ident(x):
        calls.append(x)
        return x

    ident(3)
    ident(3.0)
    ident(3)
    assert len(calls) == 2


def test_cached_accepts_callable_returning_cache():
    store = {}

    @cached(lambda: store)
    def double(x):
        return 2 * x

    assert double(21) == 42
    assert len(store) == 1


def test_cached_rejects_non_mapping_cache():
    with pytest.raises(TypeError):
        cached(42)(lambda x: x)


def test_cachedmethod_with_attribute_cache_is_per_instance():
    class Doubler:
        def __init__(self):
            self.cache = LRI()
            self.calls = 0

        @cachedmethod('cache')
        def double(self, x):
            self.calls += 1
            return 2 * x

    a, b = Doubler(), Doubler()
    assert a.double(3) == 6
    assert a.double(3) == 6
    assert b.double(3) == 6
    assert a.calls == 1
    assert b.calls == 1
    assert len(a.cache) == 1


def test_cachedmethod_unscoped_shared_cache_key():
    shared = {}

    class Thing:
        @cachedmethod(shared, scoped=False)
        def value(self, x):
            return x + 1

    assert Thing().value(1) == 2
    assert len(shared) == 1


def test_cachedproperty_computes_once_and_can_be_reset():
    class Lazy:
        def __init__(self):
            self.computed = 0

        @cachedproperty
        def expensive(self):
            self.computed += 1
            return 'result'

    obj = Lazy()
    assert obj.expensive == 'result'
    assert obj.expensive == 'result'
    assert obj.computed == 1
    del obj.expensive
    assert obj.expensive == 'result'
    assert obj.computed == 2


def test_make_cache_key_single_fast_type_is_unwrapped():
    assert make_cache_key((5,), {}) == 5
    assert make_cache_key(('abc',), {}) == 'abc'
    key = make_cache_key((1, 2), {'z': 3})
    assert key == make_cache_key((1, 2), {'z': 3})
    assert hash(key) == hash(make_cache_key((1, 2), {'z': 3}))
    assert make_cache_key((1,), {}, typed=True) != make_cache_key((1.0,), {}, typed=True)


# --------------------------------------------------------- ThresholdCounter

def test_threshold_counter_basic_counting():
    tc = ThresholdCounter(threshold=0.1)
    tc.update('aaabbc')
    assert tc.total == 6
    assert tc.get('a') == 3
    assert tc['b'] == 2
    assert tc.get('zzz') == 0
    assert 'c' in tc
    assert tc.most_common(1) == [('a', 3)]
    assert sorted(tc.elements()) == list('aaabbc')


def test_threshold_counter_update_with_mapping_and_kwargs():
    tc = ThresholdCounter(threshold=0.01)
    tc.update({'x': 3, 'y': 1}, z=2)
    assert tc.total == 6
    assert tc['x'] == 3
    assert tc['z'] == 2


@pytest.mark.parametrize('threshold', [0, 1, -0.5, 2])
def test_threshold_counter_rejects_bad_threshold(threshold):
    with pytest.raises(ValueError):
        ThresholdCounter(threshold=threshold)


def test_threshold_counter_compacts_rare_keys():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    tc.update([2] * 10)
    assert tc.get(1) == 0           # culled at the first compaction
    assert tc[2] == 10
    assert tc.get_uncommon_count() == 1
    assert tc.get_common_count() == 10
    assert tc.get_commonality() == pytest.approx(10 / 11)


def test_threshold_counter_most_common_edge_cases():
    tc = ThresholdCounter(threshold=0.1)
    tc.update('abbccc')
    assert tc.most_common() == [('c', 3), ('b', 2), ('a', 1)]
    assert tc.most_common(0) == []
    assert tc.most_common(10) == tc.most_common()
    assert tc.threshold == 0.1


def test_min_id_map_assigns_and_reuses_smallest_ids():
    class Obj:
        pass

    ids = MinIDMap()
    first, second, third = Obj(), Obj(), Obj()
    assert ids.get(first) == 0
    assert ids.get(second) == 1
    assert ids.get(first) == 0
    ids.drop(first)
    assert first not in ids
    assert ids.get(third) == 0
    assert len(ids) == 2


# ------------------------------------------------------------------ queues

QUEUE_CLASSES = [HeapPriorityQueue, SortedPriorityQueue]


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_queue_orders_by_priority_then_insertion(cls):
    pq = cls()
    pq.add('low', 0)
    pq.add('high', 2)
    pq.add('medium 1', 1)
    pq.add('medium 2', 1)
    assert len(pq) == 4
    assert pq.pop() == 'high'
    assert pq.peek() == 'medium 1'
    assert pq.pop() == 'medium 1'
    assert pq.pop() == 'medium 2'
    assert pq.pop() == 'low'
    assert len(pq) == 0


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_queue_remove_skips_task(cls):
    pq = cls()
    for name, prio in [('a', 3), ('b', 2), ('c', 1)]:
        pq.add(name, prio)
    pq.remove('a')
    assert len(pq) == 2
    assert pq.pop() == 'b'
    with pytest.raises(KeyError):
        pq.remove('a')


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_queue_readd_changes_priority(cls):
    pq = cls()
    pq.add('x', 1)
    pq.add('y', 2)
    pq.add('x', 5)
    assert len(pq) == 2
    assert pq.pop() == 'x'
    assert pq.pop() == 'y'


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_queue_empty_behaviour(cls):
    pq = cls()
    with pytest.raises(IndexError):
        pq.pop()
    with pytest.raises(IndexError):
        pq.peek()
    assert pq.pop(default='nothing') == 'nothing'
    assert pq.peek(default=None) is None


@pytest.mark.parametrize('cls', QUEUE_CLASSES)
def test_priority_queue_custom_priority_key(cls):
    pq = cls(priority_key=lambda p: p)       # lower number pops first
    pq.add('late', 10)
    pq.add('early', 1)
    assert pq.pop() == 'early'


def test_priority_queue_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        HeapPriorityQueue(bogus=True)


def test_default_priority_queue_is_sorted_variant():
    assert PriorityQueue is SortedPriorityQueue
