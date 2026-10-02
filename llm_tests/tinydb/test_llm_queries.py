"""
Unit tests for tinydb.queries: Query / where, comparison operators,
exists/matches/search/test, any/all/one_of/fragment/map, and
and/or/not combinations.
"""
import pytest

from tinydb import Query, where
from tinydb.queries import QueryInstance


@pytest.fixture
def doc():
    return {
        "name": "John",
        "age": 30,
        "tags": ["admin", "user"],
        "address": {"city": "NYC", "zip": "10001"},
        "scores": [{"subject": "math", "score": 90}, {"subject": "art", "score": 60}],
    }


def test_eq_operator(doc):
    assert (where("name") == "John")(doc) is True
    assert (where("name") == "Jane")(doc) is False


def test_ne_operator(doc):
    assert (where("name") != "Jane")(doc) is True
    assert (where("name") != "John")(doc) is False


def test_lt_le_gt_ge_operators(doc):
    assert (where("age") < 31)(doc) is True
    assert (where("age") < 30)(doc) is False
    assert (where("age") <= 30)(doc) is True
    assert (where("age") > 29)(doc) is True
    assert (where("age") > 30)(doc) is False
    assert (where("age") >= 30)(doc) is True


def test_exists_true_and_false(doc):
    assert where("name").exists()(doc) is True
    assert where("nonexistent").exists()(doc) is False


def test_matches_full_string(doc):
    assert where("name").matches(r"^J\w+$")(doc) is True
    assert where("name").matches(r"^X\w+$")(doc) is False


def test_search_substring(doc):
    assert where("name").search(r"oh")(doc) is True
    assert where("name").search(r"xyz")(doc) is False


def test_test_with_custom_function(doc):
    def is_even(value):
        return value % 2 == 0

    assert where("age").test(is_even)(doc) is True
    assert where("age").test(lambda v: v > 100)(doc) is False


def test_test_with_extra_args(doc):
    def in_range(value, low, high):
        return low <= value <= high

    assert where("age").test(in_range, 20, 40)(doc) is True
    assert where("age").test(in_range, 40, 50)(doc) is False


def test_any_with_query(doc):
    q = where("scores").any(where("score") > 80)
    assert q(doc) is True
    q2 = where("scores").any(where("score") > 1000)
    assert q2(doc) is False


def test_any_with_list(doc):
    q = where("tags").any(["admin", "editor"])
    assert q(doc) is True
    q2 = where("tags").any(["superadmin"])
    assert q2(doc) is False


def test_all_with_query(doc):
    q = where("scores").all(where("score") > 10)
    assert q(doc) is True
    q2 = where("scores").all(where("score") > 80)
    assert q2(doc) is False


def test_one_of(doc):
    assert where("name").one_of(["John", "Jane"])(doc) is True
    assert where("name").one_of(["Alice", "Bob"])(doc) is False


def test_fragment_matches_partial_document(doc):
    q = Query().fragment({"name": "John", "age": 30})
    assert q(doc) is True
    q2 = Query().fragment({"name": "John", "age": 99})
    assert q2(doc) is False


def test_noop_always_true(doc):
    assert Query().noop()(doc) is True
    assert Query().noop()({}) is True


def test_and_combination(doc):
    q = (where("name") == "John") & (where("age") == 30)
    assert q(doc) is True
    q2 = (where("name") == "John") & (where("age") == 99)
    assert q2(doc) is False


def test_or_combination(doc):
    q = (where("name") == "Nope") | (where("age") == 30)
    assert q(doc) is True
    q2 = (where("name") == "Nope") | (where("age") == 99)
    assert q2(doc) is False


def test_not_combination(doc):
    q = ~(where("name") == "John")
    assert q(doc) is False
    q2 = ~(where("name") == "Jane")
    assert q2(doc) is True


def test_complex_combination(doc):
    q = ((where("age") > 18) & (where("name") == "John")) | where("nonexistent").exists()
    assert q(doc) is True


def test_nested_path_access():
    q = where("address").city == "NYC"
    assert q({"address": {"city": "NYC"}}) is True
    assert q({"address": {"city": "LA"}}) is False


def test_query_equality_and_hash_are_stable():
    q1 = where("name") == "John"
    q2 = where("name") == "John"
    assert q1 == q2
    assert hash(q1) == hash(q2)


def test_query_instance_type_and_cacheable():
    q = where("name") == "John"
    assert isinstance(q, QueryInstance)
    assert q.is_cacheable() is True
