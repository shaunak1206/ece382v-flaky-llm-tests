import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture(autouse=True)
def seed_random():
    """Make the random eviction choice in insert() deterministic."""
    random.seed(12345)


def make_filter(capacity=1000, bucket_size=4, fingerprint_size=2,
                max_displacements=500):
    return CuckooFilter(capacity,
                        bucket_size=bucket_size,
                        fingerprint_size=fingerprint_size,
                        max_displacements=max_displacements)


# ---------- construction / bucket_size ----------

def test_default_parameters():
    cf = CuckooFilter(10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf) == 0


def test_bucket_size_attribute_matches_argument():
    cf = CuckooFilter(10, bucket_size=3)
    assert cf.bucket_size == 3


def test_one_bucket_per_capacity_slot():
    cf = CuckooFilter(7, bucket_size=2)
    assert len(cf.buckets) == 7


def test_repr_reports_state():
    cf = make_filter(capacity=12, fingerprint_size=2)
    cf.insert("hello")
    text = repr(cf)
    assert "capacity=12" in text
    assert "size=1" in text
    assert "2 byte(s)" in text


# ---------- contains ----------

def test_contains_on_empty_filter_is_false():
    cf = make_filter()
    assert cf.contains("missing") is False
    assert ("missing" in cf) is False


def test_contains_after_insert_is_true():
    cf = make_filter()
    cf.insert("apple")
    assert cf.contains("apple") is True
    assert "apple" in cf


def test_no_false_negatives_for_many_items():
    cf = make_filter(capacity=1000, bucket_size=4, fingerprint_size=2)
    items = ["item-{}".format(i) for i in range(200)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.contains(item), item


# ---------- insert ----------

def test_insert_returns_true_and_increments_size():
    cf = make_filter()
    assert cf.insert("a") is True
    assert len(cf) == 1
    assert cf.size == 1


def test_insert_multiple_distinct_items_counts_each():
    cf = make_filter()
    for i in range(10):
        assert cf.insert("x{}".format(i)) is True
    assert len(cf) == 10


def test_insert_raises_when_both_buckets_are_full_and_no_displacements():
    # capacity 1 and bucket_size 1: both candidate buckets are the same
    # single slot, and max_displacements=0 means no eviction attempts.
    cf = make_filter(capacity=1, bucket_size=1, max_displacements=0)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")


def test_failed_insert_does_not_change_size():
    cf = make_filter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert len(cf) == 1


def test_insert_raises_when_filter_is_full_after_evictions():
    # One bucket with two slots: exactly two items fit, the third must fail.
    cf = make_filter(capacity=1, bucket_size=2, max_displacements=10)
    assert cf.insert("a") is True
    assert cf.insert("b") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("c")
    assert len(cf) == 2


def test_insert_fills_bucket_to_capacity_then_raises():
    cf = make_filter(capacity=1, bucket_size=3, max_displacements=5)
    for item in ["p", "q", "r"]:
        cf.insert(item)
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("s")
    assert "full" in str(excinfo.value).lower()


# ---------- delete ----------

def test_delete_inserted_item_returns_true_and_decrements_size():
    cf = make_filter()
    cf.insert("banana")
    assert cf.delete("banana") is True
    assert len(cf) == 0
    assert cf.contains("banana") is False


def test_delete_non_inserted_item_on_empty_filter_returns_false():
    cf = make_filter()
    assert cf.delete("ghost") is False
    assert len(cf) == 0


def test_delete_does_not_affect_other_items():
    cf = make_filter()
    items = ["one", "two", "three", "four"]
    for item in items:
        cf.insert(item)
    assert cf.delete("two") is True
    assert len(cf) == 3
    for item in ["one", "three", "four"]:
        assert cf.contains(item)


def test_delete_one_of_two_duplicate_inserts_keeps_item_present():
    cf = make_filter()
    cf.insert("dup")
    cf.insert("dup")
    assert len(cf) == 2
    assert cf.delete("dup") is True
    assert len(cf) == 1
    assert cf.contains("dup") is True


def test_insert_delete_cycle_restores_empty_state():
    cf = make_filter()
    items = ["alpha", "beta", "gamma"]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0
    for item in items:
        assert cf.contains(item) is False


# ---------- CuckooFilterFullException ----------

def test_full_exception_is_an_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_full_exception_can_carry_message():
    err = CuckooFilterFullException("Filter is full.")
    assert str(err) == "Filter is full."
    with pytest.raises(CuckooFilterFullException, match="full"):
        raise err
