import pytest

from cachetools import TLRUCache, TTLCache


class FakeTimer:
    """A manually advanced clock, so tests never depend on real time."""

    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------


def test_ttl_item_expires_exactly_at_ttl():
    timer = FakeTimer()
    c = TTLCache(maxsize=2, ttl=10, timer=timer)
    c["a"] = 1
    timer.now = 9
    assert "a" in c
    assert c["a"] == 1
    timer.now = 10
    assert "a" not in c
    with pytest.raises(KeyError):
        c["a"]


def test_ttl_property_and_timer_property():
    timer = FakeTimer(5)
    c = TTLCache(maxsize=2, ttl=7, timer=timer)
    assert c.ttl == 7
    assert c.timer() == 5
    timer.now = 6
    assert c.timer() == 6


def test_ttl_len_and_currsize_exclude_expired_items():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    timer.now = 5
    c["b"] = 2
    assert len(c) == 2
    assert c.currsize == 2
    timer.now = 12
    assert len(c) == 1
    assert c.currsize == 1
    assert list(c) == ["b"]


def test_ttl_expire_returns_expired_pairs():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    timer.now = 5
    c["b"] = 2
    timer.now = 11
    assert list(c.expire()) == [("a", 1)]
    assert "a" not in c
    assert c["b"] == 2


def test_ttl_overwrite_resets_expiry_but_read_does_not():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    c["b"] = 1
    timer.now = 8
    c["a"] = 2  # expires at 18
    assert c["b"] == 1  # reading must not extend the lifetime
    timer.now = 12
    assert c["a"] == 2
    assert "b" not in c


def test_ttl_evicts_least_recently_used_when_full():
    timer = FakeTimer()
    c = TTLCache(maxsize=2, ttl=100, timer=timer)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_ttl_popitem_skips_expired_and_empty_raises():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    timer.now = 5
    c["b"] = 2
    timer.now = 11
    assert c.popitem() == ("b", 2)
    with pytest.raises(KeyError):
        c.popitem()


def test_ttl_get_pop_and_delete_on_expired_items():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    timer.now = 20
    assert c.get("a", "gone") == "gone"
    assert c.pop("b", "gone") == "gone"
    with pytest.raises(KeyError):
        c.pop("b")
    with pytest.raises(KeyError):
        del c["c"]


def test_ttl_clear_empties_cache():
    timer = FakeTimer()
    c = TTLCache(maxsize=5, ttl=10, timer=timer)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert list(c.expire()) == []
    c["z"] = 26
    assert c["z"] == 26


def test_ttl_pickle_roundtrip_preserves_items():
    import pickle

    c = TTLCache(maxsize=3, ttl=100000)
    c["a"] = 1
    c["b"] = 2
    restored = pickle.loads(pickle.dumps(c))
    assert dict(restored) == {"a": 1, "b": 2}
    assert restored.ttl == 100000
    assert restored.maxsize == 3


# ---------------------------------------------------------------------------
# TLRUCache
# ---------------------------------------------------------------------------


def test_tlru_uses_ttu_to_compute_expiry():
    timer = FakeTimer()
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now + value, timer=timer)
    c["short"] = 5
    c["long"] = 50
    timer.now = 4
    assert "short" in c
    timer.now = 5
    assert "short" not in c
    assert "long" in c
    timer.now = 50
    assert "long" not in c


def test_tlru_already_expired_item_is_not_stored():
    timer = FakeTimer(10)
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now, timer=timer)
    c["a"] = 1
    assert "a" not in c
    assert len(c) == 0


def test_tlru_expired_update_removes_existing_item():
    timer = FakeTimer()
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now + value, timer=timer)
    c["a"] = 10
    assert "a" in c
    c["a"] = 0
    assert "a" not in c
    assert len(c) == 0


def test_tlru_expire_returns_pairs_and_ttu_property():
    timer = FakeTimer()

    def ttu(key, value, now):
        return now + 10

    c = TLRUCache(maxsize=5, ttu=ttu, timer=timer)
    assert c.ttu is ttu
    c["a"] = 1
    timer.now = 5
    c["b"] = 2
    timer.now = 11
    assert list(c.expire()) == [("a", 1)]
    assert c["b"] == 2


def test_tlru_evicts_least_recently_used_when_full():
    timer = FakeTimer()
    c = TLRUCache(maxsize=2, ttu=lambda key, value, now: now + 100, timer=timer)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    c["c"] = 3
    assert "b" not in c
    assert set(c) == {"a", "c"}


def test_tlru_overwrite_replaces_expiry_and_value():
    timer = FakeTimer()
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now + 10, timer=timer)
    c["a"] = 1
    timer.now = 5
    c["a"] = 2  # now expires at 15
    timer.now = 12
    assert c["a"] == 2
    assert len(c) == 1
    timer.now = 15
    assert "a" not in c


def test_tlru_delete_missing_key_raises_keyerror():
    timer = FakeTimer()
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now + 10, timer=timer)
    with pytest.raises(KeyError):
        del c["nope"]


def test_tlru_clear_empties_cache():
    timer = FakeTimer()
    c = TLRUCache(maxsize=5, ttu=lambda key, value, now: now + 10, timer=timer)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert list(c) == []
    c["c"] = 3
    assert list(c) == ["c"]
