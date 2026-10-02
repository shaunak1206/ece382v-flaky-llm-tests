"""Tests for Query / where and the helpers in tinydb.utils."""
import re

import pytest

from tinydb import Query, TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.utils import FrozenDict, LRUCache, freeze


def test_equality_and_inequality():
    assert (where("a") == 1)({"a": 1})
    assert not (where("a") == 1)({"a": 2})
    assert (where("a") != 1)({"a": 2})
    assert not (where("a") != 1)({"a": 1})
    assert (where("tags") == ["a", "b"])({"tags": ["a", "b"]})
    assert (where("opts") == {"x": 1})({"opts": {"x": 1}})


@pytest.mark.parametrize(
    "make,doc,expected",
    [
        (lambda: where("n") < 5, {"n": 4}, True),
        (lambda: where("n") < 5, {"n": 5}, False),
        (lambda: where("n") <= 5, {"n": 5}, True),
        (lambda: where("n") <= 5, {"n": 6}, False),
        (lambda: where("n") > 5, {"n": 6}, True),
        (lambda: where("n") > 5, {"n": 5}, False),
        (lambda: where("n") >= 5, {"n": 5}, True),
        (lambda: where("n") >= 5, {"n": 4}, False),
    ],
)
def test_comparison_operators(make, doc, expected):
    assert bool(make()(doc)) is expected


def test_missing_key_never_matches():
    assert not (where("a") == 1)({})
    assert not (where("a") != 1)({})
    assert not (where("a") < 1)({"b": 0})
    assert not where("a").exists()({"b": 1})


def test_nested_paths():
    doc = {"a": {"b": {"c": 1}}, "key with space": 7}
    assert (where("a").b.c == 1)(doc)
    assert not (where("a").b.c == 2)(doc)
    assert (Query().a["b"]["c"] == 1)(doc)
    assert (Query()["key with space"] == 7)(doc)


def test_path_through_a_non_mapping_is_false():
    assert not (where("a").b == 1)({"a": 5})
    assert not (where("a").b == 1)({"a": None})


def test_exists():
    assert where("a").exists()({"a": None})
    assert where("a").b.exists()({"a": {"b": 0}})
    assert not where("a").b.exists()({"a": {}})


def test_matches_and_search_regex():
    q = where("s").matches(r"^\w+$")
    assert q({"s": "John"})
    assert not q({"s": "John Smith"})
    assert not q({"s": 123})
    assert where("s").matches("^john$", re.IGNORECASE)({"s": "JOHN"})
    s = where("s").search("ell")
    assert s({"s": "hello"})
    assert not s({"s": "help"})
    assert not s({"s": 5})


def test_custom_test_function_with_arguments():
    def divisible(value, divisor):
        return value % divisor == 0

    q = where("n").test(divisible, 3)
    assert q({"n": 9})
    assert not q({"n": 10})
    assert where("n").test(lambda v: v > 1)({"n": 2})


def test_any_with_subquery_and_with_list():
    q = where("items").any(where("qty") > 5)
    assert q({"items": [{"qty": 1}, {"qty": 9}]})
    assert not q({"items": [{"qty": 1}]})
    assert not q({"items": []})
    assert not q({"items": 5})
    tags = where("tags").any(["a", "b"])
    assert tags({"tags": ["x", "b"]})
    assert not tags({"tags": ["x", "y"]})


def test_all_with_subquery_and_with_list():
    q = where("items").all(where("qty") >= 1)
    assert q({"items": [{"qty": 1}, {"qty": 2}]})
    assert not q({"items": [{"qty": 1}, {"qty": 0}]})
    tags = where("tags").all(["a", "b"])
    assert tags({"tags": ["a", "b", "c"]})
    assert not tags({"tags": ["a"]})
    assert not tags({"tags": 3})


def test_one_of():
    q = where("color").one_of(["red", "blue"])
    assert q({"color": "red"})
    assert not q({"color": "green"})


def test_fragment():
    q = Query().fragment({"a": 1, "b": 2})
    assert q({"a": 1, "b": 2, "c": 3})
    assert not q({"a": 1})
    assert not q({"a": 1, "b": 3})
    nested = where("meta").fragment({"k": "v"})
    assert nested({"meta": {"k": "v", "z": 1}})
    assert not nested({"meta": {"k": "x"}})


def test_noop_and_map():
    assert Query().noop()({})
    assert (Query().noop() & (where("a") == 1))({"a": 1})
    assert not (Query().noop() & (where("a") == 1))({"a": 2})
    q = where("name").map(len) == 4
    assert q({"name": "John"})
    assert not q({"name": "Jo"})
    assert (where("data").map(lambda d: d["x"]) > 1)({"data": {"x": 5}})


def test_logical_combinations():
    a = where("a") == 1
    b = where("b") == 2
    assert (a & b)({"a": 1, "b": 2})
    assert not (a & b)({"a": 1, "b": 3})
    assert (a | b)({"a": 0, "b": 2})
    assert not (a | b)({"a": 0, "b": 0})
    assert (~a)({"a": 2})
    assert (~a)({"b": 1})
    assert not (~a)({"a": 1})
    c = where("c") == 3
    combined = (a | b) & ~c
    assert combined({"a": 1})
    assert not combined({"a": 1, "c": 3})
    assert not combined({"a": 0, "b": 0})


def test_empty_queries_cannot_be_used():
    with pytest.raises(RuntimeError):
        Query()({})
    with pytest.raises(ValueError):
        Query() == 1
    with pytest.raises(ValueError):
        Query().exists()


def test_query_hash_and_equality():
    assert (where("a") == 1) == (where("a") == 1)
    assert hash(where("a") == 1) == hash(where("a") == 1)
    assert not ((where("a") == 1) == (where("a") == 2))
    assert not ((where("a") == 1) == "something else")
    a, b = where("a") == 1, where("b") == 2
    assert (a & b) == (b & a)
    assert (a | b) == (b | a)
    assert not ((a & b) == (a | b))
    assert {a: "x"}[where("a") == 1] == "x"


def test_cacheable_and_uncacheable_queries():
    class Unhashable:
        __hash__ = None

    plain = where("a") == 1
    assert plain.is_cacheable()
    uncacheable = where("a").map(len) == 1
    assert not uncacheable.is_cacheable()
    assert not (uncacheable & plain).is_cacheable()
    assert not (plain | uncacheable).is_cacheable()
    assert not (~uncacheable).is_cacheable()
    assert not where("a").test(lambda v, o: True, Unhashable()).is_cacheable()
    assert (plain & plain).is_cacheable()


def test_queries_against_a_database():
    db = TinyDB(storage=MemoryStorage)
    User = Query()
    db.insert_multiple(
        [
            {"name": "John", "age": 30, "tags": ["a", "b"]},
            {"name": "Jane", "age": 22, "tags": ["b"]},
            {"name": "Bob", "age": 45, "tags": []},
        ]
    )
    assert len(db.search(User.age > 25)) == 2
    names = [d["name"] for d in db.search((User.age > 20) & User.tags.any(["a"]))]
    assert names == ["John"]
    assert db.count(User.name.matches(r"^J")) == 2
    assert db.count(~User.name.exists()) == 0
    assert len(db.search(where("name").map(len) == 4)) == 2
    db.insert({"name": "Jill", "age": 1, "tags": []})
    assert len(db.search(where("name").map(len) == 4)) == 3
    db.close()


def test_lru_cache_evicts_least_recently_used():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.get("a") == 1
    cache["c"] = 3
    assert "b" not in cache
    assert cache.lru == ["a", "c"]
    assert len(cache) == 2
    assert cache.length == 2


def test_lru_cache_mapping_behaviour():
    cache = LRUCache()
    for i in range(100):
        cache[i] = str(i)
    assert len(cache) == 100
    assert cache.get("missing") is None
    assert cache.get("missing", "dflt") == "dflt"
    with pytest.raises(KeyError):
        cache["missing"]
    del cache[0]
    assert 0 not in cache
    cache.clear()
    assert len(cache) == 0


def test_freeze_makes_nested_structures_hashable():
    frozen = freeze({"a": [1, 2, {"b": 3}], "c": {4, 5}})
    assert isinstance(frozen, FrozenDict)
    assert frozen["a"] == (1, 2, {"b": 3})
    assert frozen["c"] == frozenset({4, 5})
    assert isinstance(hash(frozen), int)
    assert freeze(5) == 5
    assert isinstance(hash(freeze(({"a": [1]},))), int)


def test_frozen_dict_is_immutable():
    fd = freeze({"a": 1})
    with pytest.raises(TypeError):
        fd["b"] = 2
    with pytest.raises(TypeError):
        del fd["a"]
    with pytest.raises(TypeError):
        fd.update({"c": 3})
    with pytest.raises(TypeError):
        fd.pop("a")
    with pytest.raises(TypeError):
        fd.clear()
