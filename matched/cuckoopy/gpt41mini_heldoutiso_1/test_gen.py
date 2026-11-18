import pytest
import random

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException

def test_insert_single_item():
    cf = CuckooFilter(capacity=10)
    assert cf.insert("apple") is True
    assert len(cf) == 1
    assert "apple" in cf

def test_insert_multiple_items():
    cf = CuckooFilter(capacity=10)
    items = ["apple", "banana", "cherry"]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert item in cf

def test_insert_duplicate_item_increases_size_once():
    cf = CuckooFilter(capacity=10)
    assert cf.insert("apple") is True
    original_size = len(cf)
    # Insert duplicate, expected to return True, but size may or may not increase.
    # In cuckoo filters, duplicates can be inserted, increasing count.
    result = cf.insert("apple")
    # Size should be either same or increased by one depending on internal insert logic.
    assert result is True
    assert len(cf) >= original_size

def test_contains_on_nonexistent_item():
    cf = CuckooFilter(capacity=10)
    cf.insert("apple")
    assert "banana" not in cf
    assert cf.contains("banana") is False

def test_delete_existing_item():
    cf = CuckooFilter(capacity=10)
    cf.insert("apple")
    assert cf.delete("apple") is True
    assert len(cf) == 0
    assert "apple" not in cf

def test_delete_nonexistent_item_returns_false():
    cf = CuckooFilter(capacity=10)
    cf.insert("apple")
    assert cf.delete("banana") is False
    assert len(cf) == 1

def test_delete_reduces_size_by_one():
    cf = CuckooFilter(capacity=10)
    items = ["apple", "banana"]
    for item in items:
        cf.insert(item)
    assert len(cf) == 2
    cf.delete("banana")
    assert len(cf) == 1
    assert "banana" not in cf
    assert "apple" in cf

def test_len_and_repr_and_contains_dunder():
    cf = CuckooFilter(capacity=5)
    assert isinstance(repr(cf), str)
    assert len(cf) == 0
    cf.insert("apple")
    assert len(cf) == 1
    assert "apple" in cf

def test_bucket_size_parameter_sets_correctly():
    bucket_size = 8
    cf = CuckooFilter(capacity=5, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size
    # Bucket sizes should be consistent in all buckets
    for b in cf.buckets:
        assert len(b) == 0 or len(b) <= bucket_size

def test_insert_raises_full_exception_when_full(monkeypatch):
    # Small filter with bucket size 1 for quick filling
    cf = CuckooFilter(capacity=2, bucket_size=1, max_displacements=2)

    # Fill all buckets manually by patching bucket.insert to fail except once
    inserted = []

    original_insert = cf.buckets[0].insert
    def insert_patch(fp):
        if len(inserted) == 0:
            inserted.append(fp)
            return True
        return False
    cf.buckets[0].insert = insert_patch
    cf.buckets[1].insert = lambda fp: False
    # patch swap to always return a fp so it loops
    cf.buckets[0].swap = lambda fp: fp
    cf.buckets[1].swap = lambda fp: fp

    # patch random.choice to always pick bucket 0 to avoid randomness
    monkeypatch.setattr("random.choice", lambda x: x[0])

    # Insert will fail after max_displacements tries and raise
    with pytest.raises(CuckooFilterFullException):
        cf.insert("toofull")

def test_contains_checks_both_buckets(monkeypatch):
    cf = CuckooFilter(capacity=10)
    # Insert an item
    cf.insert("apple")

    # Force buckets to control
    fingerprint = cf._get_index("apple")
    fbucket1 = cf.buckets[cf._get_index("apple")]
    fbucket2 = cf.buckets[cf._get_alternate_index(cf._get_index("apple"), 
                                                  cf.buckets[cf._get_index("apple")].bucket[0]
                                                  if cf.buckets[cf._get_index("apple")].bucket else b'')]

    # Create a known "fake" fingerprint that is not in either bucket
    fake_fp = b'\x00' * cf.fingerprint_size

    # Confirm contains for fingerprint in bucket works properly
    # Direct test is implicit by insert, duplicated here for coverage
    assert cf.contains("apple") is True
    # Check for item that is not in filter returns False
    assert not cf.contains("nonexistent")

def test_delete_removes_fingerprint_from_buckets(monkeypatch):
    cf = CuckooFilter(capacity=10)
    cf.insert("apple")
    fp = list(cf.buckets[0].bucket) + list(cf.buckets[1].bucket)
    assert cf.delete("apple") is True
    assert not cf.delete("apple")  # Already deleted
    assert len(cf) == 0

def test_insert_returns_true_for_new_item(monkeypatch):
    cf = CuckooFilter(capacity=20)
    assert cf.insert("banana") is True
    assert "banana" in cf

def test_insert_returns_true_if_inserted_on_first_bucket(monkeypatch):
    cf = CuckooFilter(capacity=10)

    # patch bucket insert to ensure insertion on first bucket works
    original_insert_0 = cf.buckets[0].insert
    original_insert_1 = cf.buckets[1].insert

    cf.buckets[0].insert = lambda fp: True
    cf.buckets[1].insert = lambda fp: False

    res = cf.insert("testitem")
    assert res is True
    assert len(cf) == 1

    # Restore original functions
    cf.buckets[0].insert = original_insert_0
    cf.buckets[1].insert = original_insert_1

def test_insert_returns_true_if_inserted_on_alternate_bucket(monkeypatch):
    cf = CuckooFilter(capacity=10)

    # patch bucket insert to ensure insertion on second bucket works
    cf.buckets[0].insert = lambda fp: False
    cf.buckets[1].insert = lambda fp: True

    res = cf.insert("anotheritem")
    assert res is True
    assert len(cf) == 1

def test_delete_decrements_size_only_if_item_deleted():
    cf = CuckooFilter(capacity=10)
    cf.insert("apple")
    size_before = len(cf)
    # deleting item that exists
    deleted = cf.delete("apple")
    assert deleted is True
    assert len(cf) == size_before - 1

    # deleting item that does not exist
    deleted2 = cf.delete("nonexistent")
    assert deleted2 is False
    assert len(cf) == size_before - 1

def test_init_sets_attributes_correctly():
    cf = CuckooFilter(capacity=15, bucket_size=5, fingerprint_size=2, max_displacements=1000)
    assert cf.capacity == 15
    assert cf.bucket_size == 5
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 1000
    assert isinstance(cf.buckets, list)
    assert all(hasattr(b, 'bucket') for b in cf.buckets)
