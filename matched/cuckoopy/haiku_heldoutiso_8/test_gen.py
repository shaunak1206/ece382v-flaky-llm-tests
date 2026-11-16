"""Unit tests for cuckoopy.cuckoofilter.CuckooFilter and cuckoopy.exceptions."""

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_init_stores_parameters():
    cf = CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2,
                      max_displacements=42)
    assert cf.capacity == 10
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 42


def test_init_default_parameters():
    cf = CuckooFilter(capacity=8)
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500


def test_init_creates_one_bucket_per_capacity_slot():
    cf = CuckooFilter(capacity=17)
    assert len(cf.buckets) == 17


def test_bucket_size_attribute_reflects_constructor_argument():
    cf = CuckooFilter(capacity=5, bucket_size=2)
    assert cf.bucket_size == 2


def test_bucket_size_one_is_accepted():
    cf = CuckooFilter(capacity=4, bucket_size=1)
    assert cf.bucket_size == 1
    assert cf.insert("solo") is True
    assert cf.contains("solo")


def test_initial_len_is_zero():
    cf = CuckooFilter(capacity=10)
    assert len(cf) == 0


def test_repr_contains_filter_state():
    cf = CuckooFilter(capacity=12, fingerprint_size=2)
    text = repr(cf)
    assert "capacity=12" in text
    assert "size=0" in text
    assert "fingerprint size=2" in text


# ---------------------------------------------------------------------------
# insert / contains
# ---------------------------------------------------------------------------

def test_insert_returns_true_and_increments_len():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    assert cf.insert("apple") is True
    assert len(cf) == 1


def test_inserted_item_is_contained():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    cf.insert("banana")
    assert cf.contains("banana") is True


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=50, fingerprint_size=4)
    assert cf.contains("never-inserted") is False


def test_dunder_contains_matches_contains_method():
    cf = CuckooFilter(capacity=50, fingerprint_size=4)
    cf.insert("cherry")
    assert ("cherry" in cf) is True
    assert ("cherry" in cf) == cf.contains("cherry")


def test_inserting_duplicate_increments_len_each_time():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    cf.insert("dup")
    cf.insert("dup")
    assert len(cf) == 2
    assert cf.contains("dup")


def test_bulk_insert_has_no_false_negatives():
    cf = CuckooFilter(capacity=200, bucket_size=4, fingerprint_size=2)
    items = ["item-%d" % i for i in range(300)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item), item


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_present_item_returns_true_and_decrements_len():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    cf.insert("grape")
    assert cf.delete("grape") is True
    assert len(cf) == 0


def test_delete_absent_item_on_empty_filter_returns_false():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    assert cf.delete("ghost") is False
    assert len(cf) == 0


def test_delete_one_copy_of_duplicate_keeps_other_copy():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    cf.insert("twin")
    cf.insert("twin")
    assert cf.delete("twin") is True
    assert len(cf) == 1
    assert cf.contains("twin")


def test_delete_all_inserted_items_empties_filter():
    cf = CuckooFilter(capacity=200, bucket_size=4, fingerprint_size=2)
    items = ["key-%d" % i for i in range(100)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0


# ---------------------------------------------------------------------------
# Full filter / CuckooFilterFullException
# ---------------------------------------------------------------------------

def test_insert_raises_when_filter_is_full():
    # Only capacity * bucket_size = 2 slots exist, so 10 inserts must fail.
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=2,
                      max_displacements=10)
    with pytest.raises(CuckooFilterFullException):
        for i in range(10):
            cf.insert("overflow-%d" % i)


def test_full_exception_message_mentions_full():
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=2,
                      max_displacements=10)
    with pytest.raises(CuckooFilterFullException) as excinfo:
        for i in range(10):
            cf.insert("msg-%d" % i)
    assert "full" in str(excinfo.value).lower()


def test_exception_class_is_subclass_of_exception():
    assert issubclass(CuckooFilterFullException, Exception)


def test_exception_can_be_raised_and_caught_with_message():
    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        raise CuckooFilterFullException("Insert operation failed. Filter is full.")


def test_exception_is_catchable_as_generic_exception():
    with pytest.raises(Exception):
        raise CuckooFilterFullException("generic catch")
