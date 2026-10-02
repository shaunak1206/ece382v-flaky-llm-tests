"""Tests for boltons.cacheutils and boltons.queueutils."""

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
from boltons.queueutils import (
    HeapPriorityQueue,
    PriorityQueue,
    SortedPriorityQueue,
)


# ---------------------------------------------------------------------------
# LRU / LRI
# ---------------------------------------------------------------------------


def test_lru_evicts_the_least_recently_used_key():
    cache = LRU(max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1  # touching "a" makes "b" the oldest entry
    cache["c"] = 3
    assert "b" not in cache
    assert sorted(cache) == ["a", "c"]
    assert len(cache) == 2


def test_lri_evicts_the_least_recently_inserted_key():
    cache = LRI(max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1  # reads do not refresh an LRI entry
    cache["c"] = 3
    assert "a" not in cache
    assert sorted(cache) == ["b", "c"]


@pytest.mark.parametrize("cls", [LRI, LRU])
def test_cache_hit_miss_and_soft_miss_counters(cls):
    cap_cache = cls(max_size=2)
    cap_cache["a"], cap_cache["b"] = "A", "B"
    assert dict(cap_cache) == {"a": "A", "b": "B"}
    assert [cap_cache["b"] for _ in range(3)][0] == "B"
    cap_cache["c"] = "C"
    assert cap_cache.get("a") is None
    assert (cap_cache.hit_count, cap_cache.miss_count, cap_cache.soft_miss_count) == (3, 1, 1)


@pytest.mark.parametrize("cls", [LRI, LRU])
def test_cache_constructor_validation_and_on_miss(cls):
    with pytest.raises(ValueError):
        cls(max_size=0)
    with pytest.raises(ValueError):
        cls(max_size=-5)
    with pytest.raises(TypeError):
        cls(on_miss="not callable")

    cache = cls(on_miss=lambda key: key * 2)
    assert cache[4] == 8
    assert 4 in cache
    assert cache.miss_count == 1
    assert cache[4] == 8
    assert cache.hit_count == 1


def test_lru_dict_style_mutators():
    cache = LRU(max_size=3, values={"a": 1})
    assert cache.setdefault("b", 2) == 2
    assert cache.setdefault("a", 99) == 1
    cache.update({"c": 3})
    assert dict(cache) == {"a": 1, "b": 2, "c": 3}
    assert cache.pop("a") == 1
    assert cache.pop("zzz", "dflt") == "dflt"
    with pytest.raises(KeyError):
        cache.pop("zzz")
    assert len(cache) == 2
    cache.popitem()
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    cache["x"] = 1
    assert cache["x"] == 1


def test_cache_copy_equality_and_repr():
    original = LRI(max_size=4, values=[("a", 1), ("b", 2)])
    duplicate = original.copy()
    assert duplicate == original
    assert duplicate is not original
    assert duplicate.max_size == 4
    assert original == LRU(max_size=4, values=[("a", 1), ("b", 2)])
    duplicate["z"] = 26
    assert duplicate != original
    assert repr(LRU(max_size=2, values={"a": 1})) == (
        "LRU(max_size=2, on_miss=None, values={'a': 1})"
    )


# ---------------------------------------------------------------------------
# cached / cachedmethod / cachedproperty / make_cache_key
# ---------------------------------------------------------------------------


def test_cached_function_with_lru_cache_and_kwargs():
    calls = []
    cache = LRU()

    @cached(cache)
    def add(a, b=0):
        calls.append((a, b))
        return a + b

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert add(1, b=3) == 4
    assert add(2) == 2
    assert calls == [(1, 2), (1, 3), (2, 0)]
    assert len(cache) == 3
    assert repr(add).startswith("CachedFunction(func=")


def test_cached_typed_callable_cache_and_bad_cache():
    calls = []

    @cached({}, typed=True)
    def ident(x):
        calls.append(x)
        return x

    ident(3)
    ident(3.0)
    ident(3)
    assert calls == [3, 3.0]

    store = {}

    @cached(lambda: store)
    def square(x):
        return x * x

    assert square(4) == 16
    assert 4 in store

    with pytest.raises(TypeError):
        cached(42)(lambda x: x)


class Lowerer:
    def __init__(self):
        self.cache = LRI()
        self.calls = 0

    @cachedmethod("cache")
    def lower(self, text):
        self.calls += 1
        return text.lower()


def test_cachedmethod_uses_instance_attribute_cache_and_scopes_shared_caches():
    shared = {}

    class Adder:
        def __init__(self, n):
            self.n = n

        @cachedmethod(shared)
        def add(self, x):
            return self.n + x

    lowerer = Lowerer()
    assert lowerer.lower("WOW WHO COULD GUESS CACHING COULD BE SO NEAT") == (
        "wow who could guess caching could be so neat"
    )
    lowerer.lower("WOW WHO COULD GUESS CACHING COULD BE SO NEAT")
    assert lowerer.calls == 1
    assert len(lowerer.cache) == 1

    # a shared cache is scoped per instance by default
    assert Adder(1).add(1) == 2
    assert Adder(10).add(1) == 11
    assert len(shared) == 2


class Lazy:
    def __init__(self):
        self.count = 0

    @cachedproperty
    def value(self):
        self.count += 1
        return 42


def test_cachedproperty_computes_once_until_deleted():
    obj = Lazy()
    assert obj.value == 42
    assert obj.value == 42
    assert obj.count == 1
    del obj.value
    assert obj.value == 42
    assert obj.count == 2
    assert isinstance(Lazy.value, cachedproperty)


def test_make_cache_key_is_order_insensitive_for_kwargs():
    k1 = make_cache_key((1,), {"a": 1, "b": 2})
    k2 = make_cache_key((1,), {"b": 2, "a": 1})
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert make_cache_key(("x",), {}) == "x"
    assert make_cache_key((1, 2), {}) != make_cache_key((2, 1), {})
    assert make_cache_key((3,), {}, typed=True) != make_cache_key((3.0,), {}, typed=True)


# ---------------------------------------------------------------------------
# ThresholdCounter
# ---------------------------------------------------------------------------


def test_threshold_counter_docstring_scenario():
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11


def test_threshold_counter_counts_ranking_and_validation():
    tc = ThresholdCounter()
    tc.update("abracadabra")
    assert tc["a"] == 5
    assert tc.get("z") == 0
    assert tc.get("z", -1) == -1
    assert tc.total == 11
    assert sorted(tc.keys()) == ["a", "b", "c", "d", "r"]
    assert tc.most_common(1) == [("a", 5)]
    assert tc.most_common(2) == [("a", 5), ("b", 2)]
    assert tc.most_common(0) == []
    assert tc.get_common_count() == 11
    assert tc.get_uncommon_count() == 0
    assert tc.get_commonality() == 1.0

    tc.update({"x": 3})
    tc.update({"x": 1}, y=2)
    assert tc["x"] == 4
    assert tc["y"] == 2

    for bad in (0, 1, -0.5, 2):
        with pytest.raises(ValueError):
            ThresholdCounter(threshold=bad)


# ---------------------------------------------------------------------------
# Priority queues
# ---------------------------------------------------------------------------


def test_priority_queue_docstring_example():
    pq = PriorityQueue()
    pq.add("low priority task", 0)
    pq.add("high priority task", 2)
    pq.add("medium priority task 1", 1)
    pq.add("medium priority task 2", 1)
    assert len(pq) == 4
    assert pq.pop() == "high priority task"
    assert pq.peek() == "medium priority task 1"
    assert len(pq) == 3


@pytest.mark.parametrize("pq_cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_priority_queue_orders_by_priority_then_insertion(pq_cls):
    pq = pq_cls()
    pq.add("low", 0)
    pq.add("high", 2)
    pq.add("mid1", 1)
    pq.add("mid2", 1)
    assert pq.peek() == "high"
    assert [pq.pop() for _ in range(4)] == ["high", "mid1", "mid2", "low"]
    assert len(pq) == 0


@pytest.mark.parametrize("pq_cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_priority_queue_readd_changes_priority_and_remove(pq_cls):
    pq = pq_cls()
    pq.add("x", 1)
    pq.add("y", 2)
    pq.add("x", 3)
    assert len(pq) == 2
    assert pq.pop() == "x"
    pq.add("z", 5)
    assert len(pq) == 2
    pq.remove("z")
    assert len(pq) == 1
    with pytest.raises(KeyError):
        pq.remove("missing")
    assert pq.pop() == "y"


@pytest.mark.parametrize("pq_cls", [HeapPriorityQueue, SortedPriorityQueue])
def test_priority_queue_empty_behavior_custom_key_and_bad_kwargs(pq_cls):
    pq = pq_cls()
    assert pq.peek(default="none") == "none"
    assert pq.pop(default=None) is None
    with pytest.raises(IndexError):
        pq.peek()
    with pytest.raises(IndexError):
        pq.pop()

    # lower numbers first with a custom priority_key
    min_first = pq_cls(priority_key=lambda p: p)
    min_first.add("late", 10)
    min_first.add("soon", 1)
    assert min_first.pop() == "soon"

    # a missing priority counts as zero
    default = pq_cls()
    default.add("none")
    default.add("p1", 1)
    assert default.pop() == "p1"
    assert default.pop() == "none"

    with pytest.raises(TypeError):
        pq_cls(bogus=1)
