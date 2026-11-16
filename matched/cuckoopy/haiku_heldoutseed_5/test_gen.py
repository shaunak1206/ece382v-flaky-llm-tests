"""
Unit tests for cuckoopy.cuckoofilter.CuckooFilter and
cuckoopy.exceptions.CuckooFilterFullException.
"""

import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


# ---------------------------------------------------------------------------
# Constructor and configuration
# ---------------------------------------------------------------------------

def test_default_parameters():
    cf = CuckooFilter(capacity=10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500


def test_custom_parameters_are_stored():
    cf = CuckooFilter(capacity=7, bucket_size=2, fingerprint_size=3,
                      max_displacements=42)
    assert cf.capacity == 7
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 3
    assert cf.max_displacements == 42


def test_number_of_buckets_matches_capacity():
    cf = CuckooFilter(capacity=13, bucket_size=3)
    assert len(cf.buckets) == 13


def test_new_filter_is_empty():
    cf = CuckooFilter(capacity=8)
    assert len(cf) == 0
    assert cf.size == 0


def test_repr_reports_capacity_size_and_fingerprint():
    cf = CuckooFilter(capacity=5, fingerprint_size=2)
    cf.insert("item")
    text = repr(cf)
    assert "capacity=5" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text


def test_bucket_size_limits_slots_in_single_bucket_filter():
    # With capacity=1 every item maps to the same bucket, so exactly
    # bucket_size entries fit before the filter reports it is full.
    cf = CuckooFilter(capacity=1, bucket_size=3, fingerprint_size=4)
    assert cf.insert("a") is True
    assert cf.insert("b") is True
    assert cf.insert("c") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("d")
    assert len(cf) == 3


# ---------------------------------------------------------------------------
# insert / contains
# ---------------------------------------------------------------------------

def test_insert_returns_true_and_increments_len():
    cf = CuckooFilter(capacity=16)
    assert cf.insert("hello") is True
    assert len(cf) == 1


def test_inserted_item_is_contained():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("hello")
    assert cf.contains("hello") is True


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=16, fingerprint_size=4)
    assert cf.contains("anything") is False


def test_contains_absent_item_with_wide_fingerprint():
    cf = CuckooFilter(capacity=64, fingerprint_size=4)
    cf.insert("apple")
    cf.insert("cherry")
    # A 4-byte fingerprint makes a false positive here astronomically unlikely.
    assert cf.contains("definitely-not-inserted-xyz") is False


def test_contains_does_not_change_len():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("x")
    before = len(cf)
    cf.contains("x")
    cf.contains("y")
    assert len(cf) == before


def test_dunder_contains_matches_contains_method():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("present")
    assert ("present" in cf) is True
    assert ("present" in cf) == cf.contains("present")


def test_duplicate_inserts_are_counted():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=2)
    cf.insert("dup")
    cf.insert("dup")
    assert len(cf) == 2
    assert cf.contains("dup") is True


def test_no_false_negatives_for_many_items():
    random.seed(12345)
    cf = CuckooFilter(capacity=256, bucket_size=4, fingerprint_size=2)
    items = ["item-{}".format(i) for i in range(100)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item) is True


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_present_item_returns_true_and_decrements_len():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("gone")
    assert cf.delete("gone") is True
    assert len(cf) == 0
    assert cf.contains("gone") is False


def test_delete_absent_item_returns_false():
    cf = CuckooFilter(capacity=16, fingerprint_size=4)
    cf.insert("present")
    assert cf.delete("missing-item-xyz") is False
    assert len(cf) == 1


def test_delete_from_empty_filter_returns_false():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    assert cf.delete("nothing") is False
    assert len(cf) == 0


def test_delete_one_of_duplicate_inserts_keeps_item():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=2)
    cf.insert("twice")
    cf.insert("twice")
    assert cf.delete("twice") is True
    assert len(cf) == 1
    assert cf.contains("twice") is True
    assert cf.delete("twice") is True
    assert len(cf) == 0
    assert cf.contains("twice") is False


# ---------------------------------------------------------------------------
# Full filter behaviour and exceptions
# ---------------------------------------------------------------------------

def test_insert_into_full_filter_raises():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=5)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")


def test_failed_insert_leaves_len_unchanged():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=5)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert len(cf) == 1


def test_full_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_full_exception_message_mentions_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=5)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException, match="full"):
        cf.insert("second")


def test_full_exception_can_be_raised_and_caught_directly():
    with pytest.raises(CuckooFilterFullException) as excinfo:
        raise CuckooFilterFullException("custom message")
    assert str(excinfo.value) == "custom message"
