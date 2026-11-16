"""
Unit tests for cuckoopy.cuckoofilter.CuckooFilter and
cuckoopy.exceptions.CuckooFilterFullException.
"""

import random

import pytest

from cuckoopy import exceptions
from cuckoopy.cuckoofilter import CuckooFilter


# A 4-byte fingerprint makes false positives for absent items practically
# impossible in these small tests, so membership assertions stay reliable.
WIDE_FINGERPRINT = 4


@pytest.fixture
def cf():
    random.seed(12345)
    return CuckooFilter(capacity=100, bucket_size=4,
                        fingerprint_size=WIDE_FINGERPRINT)


# --- Constructor / attributes -------------------------------------------------

def test_init_stores_constructor_arguments():
    f = CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2,
                     max_displacements=42)
    assert f.capacity == 10
    assert f.bucket_size == 3
    assert f.fingerprint_size == 2
    assert f.max_displacements == 42


def test_init_defaults():
    f = CuckooFilter(capacity=5)
    assert f.bucket_size == 4
    assert f.fingerprint_size == 1
    assert f.max_displacements == 500
    assert len(f) == 0


def test_init_creates_one_bucket_per_capacity_slot():
    f = CuckooFilter(capacity=7, bucket_size=2)
    assert len(f.buckets) == 7


def test_bucket_size_attribute_reflects_argument():
    for size in (1, 2, 8):
        f = CuckooFilter(capacity=4, bucket_size=size)
        assert f.bucket_size == size


def test_bucket_size_default_is_four():
    assert CuckooFilter(capacity=4).bucket_size == 4


# --- insert -------------------------------------------------------------------

def test_insert_returns_true_and_increments_size(cf):
    assert cf.insert("apple") is True
    assert len(cf) == 1


def test_inserted_item_is_contained(cf):
    cf.insert("banana")
    assert cf.contains("banana")
    assert "banana" in cf


def test_insert_multiple_distinct_items_increments_len(cf):
    items = ["item-{}".format(i) for i in range(20)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item)


def test_insert_duplicate_item_counts_each_insertion(cf):
    cf.insert("dup")
    cf.insert("dup")
    assert len(cf) == 2


def test_insert_full_filter_raises_exception():
    # capacity=1, bucket_size=1: only one slot exists, so the second insert
    # must displace repeatedly and eventually fail.
    f = CuckooFilter(capacity=1, bucket_size=1,
                     fingerprint_size=WIDE_FINGERPRINT, max_displacements=5)
    assert f.insert("first") is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("second")


def test_failed_insert_does_not_change_size():
    f = CuckooFilter(capacity=1, bucket_size=1,
                     fingerprint_size=WIDE_FINGERPRINT, max_displacements=5)
    f.insert("first")
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("second")
    assert len(f) == 1


def test_insert_fills_reasonably_loaded_filter_without_error():
    random.seed(7)
    f = CuckooFilter(capacity=100, bucket_size=4,
                     fingerprint_size=WIDE_FINGERPRINT)
    # 50 items in 400 slots (12.5% load) must always fit.
    for i in range(50):
        assert f.insert("key-{}".format(i)) is True
    assert len(f) == 50


# --- contains -----------------------------------------------------------------

def test_contains_empty_filter_returns_false(cf):
    assert cf.contains("nothing") is False
    assert ("nothing" in cf) is False


def test_contains_absent_item_returns_false(cf):
    cf.insert("present")
    assert cf.contains("absent-xyz") is False


def test_contains_is_idempotent(cf):
    cf.insert("stable")
    assert cf.contains("stable")
    assert cf.contains("stable")
    assert len(cf) == 1


# --- delete -------------------------------------------------------------------

def test_delete_inserted_item_returns_true_and_decrements_size(cf):
    cf.insert("gone")
    assert cf.delete("gone") is True
    assert len(cf) == 0
    assert cf.contains("gone") is False


def test_delete_absent_item_returns_false_and_keeps_size(cf):
    cf.insert("kept")
    assert cf.delete("never-inserted") is False
    assert len(cf) == 1
    assert cf.contains("kept")


def test_delete_on_empty_filter_returns_false(cf):
    assert cf.delete("anything") is False
    assert len(cf) == 0


def test_delete_twice_second_call_returns_false(cf):
    cf.insert("once")
    assert cf.delete("once") is True
    assert cf.delete("once") is False
    assert len(cf) == 0


def test_delete_one_of_two_duplicate_insertions_keeps_item(cf):
    cf.insert("dup")
    cf.insert("dup")
    assert cf.delete("dup") is True
    assert len(cf) == 1
    assert cf.contains("dup")


def test_insert_after_delete_reuses_capacity():
    f = CuckooFilter(capacity=1, bucket_size=1,
                     fingerprint_size=WIDE_FINGERPRINT, max_displacements=5)
    f.insert("a")
    assert f.delete("a") is True
    assert f.insert("b") is True
    assert f.contains("b")
    assert len(f) == 1


# --- CuckooFilterFullException ------------------------------------------------

def test_full_exception_is_exception_subclass():
    assert issubclass(exceptions.CuckooFilterFullException, Exception)


def test_full_exception_can_be_raised_with_message():
    with pytest.raises(exceptions.CuckooFilterFullException) as excinfo:
        raise exceptions.CuckooFilterFullException("Filter is full.")
    assert "Filter is full." in str(excinfo.value)


def test_insert_full_exception_message_mentions_full():
    f = CuckooFilter(capacity=1, bucket_size=1,
                     fingerprint_size=WIDE_FINGERPRINT, max_displacements=3)
    f.insert("x")
    with pytest.raises(exceptions.CuckooFilterFullException) as excinfo:
        f.insert("y")
    assert "full" in str(excinfo.value).lower()
