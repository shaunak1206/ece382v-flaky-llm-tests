import pytest

from cachetools import FIFOCache, LFUCache, LRUCache, RRCache


# ---------------------------------------------------------------- FIFO

def test_fifo_evicts_oldest_inserted_key():
    c = FIFOCache(maxsize=3)
    for i in range(1, 5):
        c[i] = i * 10
    assert len(c) == 3
    assert 1 not in c
    assert set(c) == {2, 3, 4}


def test_fifo_access_does_not_change_eviction_order():
    c = FIFOCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "a" not in c
    assert "b" in c
    assert "c" in c


def test_fifo_popitem_returns_first_inserted_pair():
    c = FIFOCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert c.popitem() == ("a", 1)
    assert c.popitem() == ("b", 2)
    assert len(c) == 1


def test_fifo_popitem_on_empty_raises_keyerror():
    c = FIFOCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


# ----------------------------------------------------------------- LRU

def test_lru_evicts_least_recently_used_after_access():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert c["a"] == 1
    assert c["c"] == 3


def test_lru_get_counts_as_use():
    c = LRUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    assert c.get("a") == 1
    c["c"] = 3
    assert "b" not in c
    assert "a" in c


def test_lru_popitem_returns_least_recently_used_pair():
    c = LRUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"]
    assert c.popitem() == ("b", 2)
    assert c.popitem() == ("c", 3)
    assert c.popitem() == ("a", 1)


def test_lru_popitem_on_empty_raises_keyerror():
    c = LRUCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


def test_lru_clear_empties_cache_and_resets_currsize():
    c = LRUCache(maxsize=5, getsizeof=len)
    c["a"] = "xx"
    c["b"] = "yyy"
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0
    assert "a" not in c
    c["z"] = "ok"
    assert c["z"] == "ok"


# ----------------------------------------------------------------- LFU

def test_lfu_evicts_least_frequently_used_key():
    c = LFUCache(maxsize=2)
    c["a"] = 1
    c["b"] = 2
    for _ in range(3):
        assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert "a" in c
    assert "c" in c


def test_lfu_popitem_returns_least_frequently_used_pair():
    c = LFUCache(maxsize=3)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    c["a"]
    c["a"]
    c["c"]
    assert c.popitem() == ("b", 2)
    assert "b" not in c


def test_lfu_popitem_on_empty_raises_keyerror():
    c = LFUCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()


# ------------------------------------------------------------------ RR

def test_rr_exposes_choice_function():
    c = RRCache(maxsize=2, choice=min)
    assert c.choice is min


def test_rr_evicts_key_selected_by_choice():
    c = RRCache(maxsize=2, choice=min)
    c[3] = "three"
    c[1] = "one"
    c[2] = "two"
    assert 1 not in c
    assert set(c) == {2, 3}
    assert c[2] == "two"
    assert c[3] == "three"


def test_rr_default_choice_never_exceeds_maxsize():
    c = RRCache(maxsize=3)
    for i in range(10):
        c[i] = i * i
    assert len(c) == 3
    assert c.currsize == 3
    for k in c:
        assert c[k] == k * k


def test_rr_popitem_returns_existing_pair_and_empty_raises():
    c = RRCache(maxsize=3)
    with pytest.raises(KeyError):
        c.popitem()
    c["a"] = 1
    c["b"] = 2
    key, value = c.popitem()
    assert (key, value) in {("a", 1), ("b", 2)}
    assert key not in c
    assert len(c) == 1
