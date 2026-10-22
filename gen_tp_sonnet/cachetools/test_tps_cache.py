import pytest

import cachetools
from cachetools import Cache


def test_new_cache_reports_maxsize_and_zero_currsize():
    c = Cache(maxsize=5)
    assert c.maxsize == 5
    assert c.currsize == 0
    assert len(c) == 0


def test_setitem_getitem_roundtrip():
    c = Cache(maxsize=5)
    c["a"] = 1
    c["b"] = [1, 2]
    assert c["a"] == 1
    assert c["b"] == [1, 2]


def test_len_iter_and_contains():
    c = Cache(maxsize=5)
    c[1] = "x"
    c[2] = "y"
    assert len(c) == 2
    assert set(iter(c)) == {1, 2}
    assert 1 in c
    assert 3 not in c


def test_delitem_removes_key_and_reduces_currsize():
    c = Cache(maxsize=5)
    c["a"] = 1
    c["b"] = 2
    del c["a"]
    assert "a" not in c
    assert c.currsize == 1
    with pytest.raises(KeyError):
        del c["a"]


def test_getitem_missing_raises_keyerror():
    c = Cache(maxsize=2)
    with pytest.raises(KeyError):
        c["missing"]


def test_get_returns_default_or_value():
    c = Cache(maxsize=2)
    c["a"] = 1
    assert c.get("a") == 1
    assert c.get("a", 99) == 1
    assert c.get("b") is None
    assert c.get("b", 99) == 99


def test_pop_returns_value_and_removes_key():
    c = Cache(maxsize=3)
    c["a"] = 10
    assert c.pop("a") == 10
    assert "a" not in c
    assert len(c) == 0


def test_pop_missing_with_default_returns_default():
    c = Cache(maxsize=3)
    assert c.pop("nope", "dflt") == "dflt"
    assert c.pop("nope", None) is None


def test_pop_missing_without_default_raises_keyerror():
    c = Cache(maxsize=3)
    with pytest.raises(KeyError):
        c.pop("nope")


def test_setdefault_inserts_only_when_missing():
    c = Cache(maxsize=3)
    assert c.setdefault("a", 1) == 1
    assert c["a"] == 1
    assert c.setdefault("a", 2) == 1
    assert c["a"] == 1
    assert c.setdefault("b") is None
    assert "b" in c


def test_getsizeof_controls_currsize():
    plain = Cache(maxsize=10)
    assert plain.getsizeof("anything at all") == 1
    plain["a"] = "xxxxx"
    assert plain.currsize == 1

    sized = Cache(maxsize=10, getsizeof=len)
    sized["a"] = "xx"
    sized["b"] = "yyy"
    assert sized.getsizeof("abcd") == 4
    assert sized.currsize == 5
    assert len(sized) == 2


def test_value_too_large_raises_valueerror():
    c = Cache(maxsize=3, getsizeof=len)
    with pytest.raises(ValueError):
        c["big"] = "abcd"
    assert "big" not in c
    assert c.currsize == 0


def test_overwrite_adjusts_currsize():
    c = Cache(maxsize=10, getsizeof=len)
    c["a"] = "xxxxx"
    assert c.currsize == 5
    c["a"] = "x"
    assert c.currsize == 1
    assert c["a"] == "x"
    assert len(c) == 1


def test_subclass_missing_hook_is_used_by_getitem():
    class Doubling(Cache):
        def __missing__(self, key):
            value = key * 2
            self[key] = value
            return value

    c = Doubling(maxsize=5)
    assert 2 not in c
    assert c[2] == 4
    assert 2 in c
    assert c[3] == 6
    assert len(c) == 2
