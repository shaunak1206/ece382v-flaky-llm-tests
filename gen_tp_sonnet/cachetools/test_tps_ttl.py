import pytest

from cachetools import TLRUCache, TTLCache


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


@pytest.fixture
def clock():
    return Clock()


def test_ttl_property_returns_configured_value(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    assert c.ttl == 10
    assert c.maxsize == 4


def test_item_available_before_expiry(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(5)
    assert "a" in c
    assert c["a"] == 1


def test_expired_item_raises_keyerror(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(11)
    with pytest.raises(KeyError):
        c["a"]


def test_expired_item_not_contained(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    clock.advance(11)
    assert "a" not in c
    assert "b" not in c


def test_expire_returns_expired_pairs_and_removes_them(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    clock.advance(11)
    expired = sorted(c.expire())
    assert expired == [("a", 1), ("b", 2)]
    assert len(c) == 0
    assert c.currsize == 0


def test_expire_with_explicit_time(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    assert list(c.expire(5)) == []
    assert "a" in c
    assert list(c.expire(100)) == [("a", 1)]
    assert len(c) == 0


def test_reinserting_key_resets_its_ttl(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(6)
    c["a"] = 2
    clock.advance(6)
    assert "a" in c
    assert c["a"] == 2


def test_reading_item_does_not_extend_ttl(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(6)
    assert c["a"] == 1
    clock.advance(5)
    assert "a" not in c


def test_maxsize_evicts_least_recently_used(clock):
    c = TTLCache(maxsize=2, ttl=100, timer=clock)
    c["a"] = 1
    c["b"] = 2
    c["a"]
    c["c"] = 3
    assert "b" not in c
    assert "a" in c
    assert "c" in c


def test_popitem_returns_least_recently_used_pair(clock):
    c = TTLCache(maxsize=4, ttl=100, timer=clock)
    c["a"] = 1
    c["b"] = 2
    c["a"]
    assert c.popitem() == ("b", 2)
    assert "b" not in c


def test_popitem_skips_expired_items(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    clock.advance(6)
    c["b"] = 2
    clock.advance(6)
    assert c.popitem() == ("b", 2)


def test_ttl_clear_empties_cache(clock):
    c = TTLCache(maxsize=4, ttl=10, timer=clock)
    c["a"] = 1
    c["b"] = 2
    c.clear()
    assert len(c) == 0
    assert c.currsize == 0
    assert "a" not in c


def test_tlru_exposes_ttu_function(clock):
    def ttu(key, value, now):
        return now + value

    c = TLRUCache(maxsize=4, ttu=ttu, timer=clock)
    assert c.ttu is ttu


def test_tlru_uses_per_item_expiry_time(clock):
    def ttu(key, value, now):
        return now + value

    c = TLRUCache(maxsize=4, ttu=ttu, timer=clock)
    c["short"] = 5
    c["long"] = 50
    clock.advance(10)
    assert "short" not in c
    assert "long" in c
    assert c["long"] == 50
