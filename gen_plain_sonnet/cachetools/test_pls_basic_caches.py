import pytest

from cachetools import Cache, FIFOCache, LFUCache, LRUCache, RRCache

# ---------------------------------------------------------------------------
# Cache (base class)
# ---------------------------------------------------------------------------


def test_cache_negative_maxsize_raises():
    with pytest.raises(ValueError):
        Cache(maxsize=-1)


def test_cache_set_get_delete():
    c = Cache(maxsize=3)
    c["a"] = 1
    assert c["a"] == 1
    assert "a" in c
    assert len(c) == 1
    del c["a"]
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_cache_maxsize_and_currsize_properties():
    c = Cache(maxsize=5)
    assert c.maxsize == 5
    assert c.currsize == 0
    c["x"] = "y"
    assert c.currsize == 1
    assert c.maxsize == 5


def test_cache_getsizeof_tracks_currsize():
    c = Cache(maxsize=10, getsizeof=len)
    c["a"] = "xxx"
    assert c.currsize == 3
    c["b"] = "yy"
    assert c.currsize == 5
    del c["a"]
    assert c.currsize == 2


def test_cache_value_too_large_raises_and_not_stored():
    c = Cache(maxsize=5, getsizeof=len)
    with pytest.raises(ValueError):
        c["k"] = "123456"
    assert "k" not in c
    assert c.currsize == 0


def test_cache_negative_item_size_raises():
    c = Cache(maxsize=5, getsizeof=lambda value: -1)
    with pytest.raises(ValueError):
        c["k"] = "v"


def test_cache_overwrite_adjusts_currsize():
    c = Cache(maxsize=10, getsizeof=len)
    c["a"] = "xxx"
    c["a"] = "x"
    assert c.currsize == 1
    assert c["a"] == "x"
    assert len(c) == 1


def test_cache_repr():
    c = Cache(maxsize=3)
    c["a"] = 1
    assert repr(c) == "Cache({'a': 1}, maxsize=3, currsize=1)"


def test_cache_get_returns_default_for_missing_key():
    c = Cache(maxsize=3)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("b", 42) == 42


def test_cache_pop_with_and_without_default():
    c = Cache(maxsize=3)
    c["a"] = 1
    assert c.pop("a") == 1
    assert "a" not in c
    assert c.pop("a", "dflt") == "dflt"
    with pytest.raises(KeyError):
        c.pop("a")


def test_cache_setdefault():
    c = Cache(maxsize=3)
    assert c.setdefault("a", 1) == 1
    assert c.setdefault("a", 2) == 1
    assert c["a"] == 1


def test_cache_missing_hook_is_used_by_getitem_only():
    class Doubler(Cache):
        def __missing__(self, key):
            value = key * 2
            self[key] = value
            return value

    c = Doubler(maxsize=5)
    assert c[2] == 4
    assert 2 in c
    assert c.get(7) is None
    assert 7 not in c


def test_cache_clear_resets_size():
    c = Cache(maxsize=10, getsizeof=len)
    c["a"] = "xx"
    c["b"] = "yyy"
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0
    assert list(c) == []


def test_cache_popitem_returns_first_inserted_pair():
    c = Cache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert "a" not in c


# ---------------------------------------------------------------------------
# FIFOCache
# ---------------------------------------------------------------------------


def test_fifo_evicts_oldest_inserted():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert set(c) == {"b", "c"}


def test_fifo_access_does_not_change_eviction_order():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "a" not in c
    assert set(c) == {"b", "c"}


def test_fifo_update_moves_key_to_end():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 10
    assert c["c"] == 3


def test_fifo_popitem_returns_first_inserted():
    c = FIFOCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    assert c.popitem() == ("a", 1)
    assert c.popitem() == ("b", 2)


def test_fifo_popitem_empty_raises_keyerror():
    c = FIFOCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


def test_fifo_getsizeof_evicts_multiple_items():
    c = FIFOCache(maxsize=5, getsizeof=len)
    c["a"] = "11"
    c["b"] = "22"
    c["c"] = "3333"
    assert list(c) == ["c"]
    assert c.currsize == 4


def test_fifo_update_with_larger_value_evicts_others():
    c = FIFOCache(maxsize=5, getsizeof=len)
    c["a"] = "11"
    c["b"] = "22"
    c["a"] = "1111"
    assert c["a"] == "1111"
    assert c.currsize == 4
    assert len(c) == 1


# ---------------------------------------------------------------------------
# LFUCache
# ---------------------------------------------------------------------------


def test_lfu_evicts_least_frequently_used():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_lfu_get_counts_as_use():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c.get("a") == 1
    c["c"] = 3
    assert "b" not in c
    assert "a" in c


def test_lfu_overwrite_counts_as_use():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 100
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 100


def test_lfu_popitem_returns_least_used():
    c = LFUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"]
    c["a"]
    c["b"]
    assert c.popitem() == ("c", 3)
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("a", 1)


def test_lfu_popitem_empty_raises_keyerror():
    c = LFUCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


def test_lfu_delete_then_insert_keeps_consistent():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    c["c"] = 3
    assert set(c) == {"b", "c"}
    assert len(c) == 2


def test_lfu_clear_then_reuse():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["a"]
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()
    c["b"] = 2
    assert c.popitem() == ("b", 2)


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


def test_lru_evicts_least_recently_used():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert set(c) == {"b", "c"}


def test_lru_getitem_refreshes_entry():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert "a" in c


def test_lru_contains_does_not_refresh_entry():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert "a" in c
    c["c"] = 3
    assert "a" not in c


def test_lru_setitem_refreshes_entry():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    c["a"] = 10
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 10


def test_lru_popitem_order_follows_usage():
    c = LRUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"]
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("c", 3)
    assert c.popitem() == ("a", 1)


def test_lru_popitem_empty_raises_keyerror():
    c = LRUCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_clear_then_reuse():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()
    c["z"] = 26
    assert c.popitem() == ("z", 26)


# ---------------------------------------------------------------------------
# RRCache
# ---------------------------------------------------------------------------


def test_rr_custom_choice_selects_victim():
    c = RRCache(maxsize=2, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert set(c) == {"b", "c"}


def test_rr_choice_picking_last_key():
    c = RRCache(maxsize=2, choice=lambda seq: seq[-1])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert set(c) == {"a", "c"}


def test_rr_choice_property_returns_function():
    def pick(seq):
        return seq[0]

    c = RRCache(maxsize=2, choice=pick)
    assert c.choice is pick


def test_rr_popitem_empty_raises_keyerror():
    c = RRCache(maxsize=2, choice=lambda seq: seq[0])
    with pytest.raises(KeyError):
        c.popitem()


def test_rr_default_choice_respects_maxsize():
    c = RRCache(maxsize=3)
    for i in range(20):
        c[i] = i
    assert len(c) == 3
    assert c.currsize == 3
    for key in c:
        assert c[key] == key


def test_rr_delete_keeps_remaining_keys_consistent():
    c = RRCache(maxsize=3, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    del c["a"]
    assert set(c) == {"b", "c"}
    c["d"] = 4
    assert set(c) == {"b", "c", "d"}
    c["e"] = 5
    assert len(c) == 3


def test_rr_clear_then_reuse():
    c = RRCache(maxsize=2, choice=lambda seq: seq[0])
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    with pytest.raises(KeyError):
        c.popitem()
    c["x"] = 1
    assert c.popitem() == ("x", 1)
