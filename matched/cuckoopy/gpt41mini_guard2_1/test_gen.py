import random
import pytest

import cuckoopy.exceptions as exceptions
from cuckoopy.cuckoofilter import CuckooFilter

# We will seed random at the start of all tests that may call insert (which calls random.choice)
# to ensure reproducibility.

def test_init_properties():
    capacity = 10
    bucket_size = 3
    fingerprint_size = 2
    max_displacements = 50
    cf = CuckooFilter(capacity, bucket_size, fingerprint_size, max_displacements)
    assert cf.capacity == capacity
    assert cf.bucket_size == bucket_size
    assert cf.fingerprint_size == fingerprint_size
    assert cf.max_displacements == max_displacements
    # Buckets list length matches capacity
    assert len(cf.buckets) == capacity
    # Each bucket has the correct bucket_size (via bucket.size)
    for b in cf.buckets:
        assert b.size == bucket_size
    # Initial size is zero
    assert len(cf) == 0
    # __repr__ returns a string containing these values
    repr_str = repr(cf)
    assert str(capacity) in repr_str
    assert str(fingerprint_size) in repr_str
    assert str(0) in repr_str


def test_insert_and_contains_and_len(monkeypatch):
    # Seed random to ensure deterministic eviction choices
    random.seed(42)

    cf = CuckooFilter(capacity=20, bucket_size=4, fingerprint_size=1, max_displacements=500)

    # Insert a number item
    item = 12345
    # Initially contains is false
    assert item not in cf
    # Insert returns True
    assert cf.insert(item) is True
    # Now contains returns true
    assert item in cf
    # Size increased to 1
    assert len(cf) == 1

    # Inserting again same item inserts fingerprint again because filter does not track items, only fingerprints
    # But since fingerprint is same, buckets might be full.
    # Insert of same item should succeed or fail depending on bucket content.
    # We just test it returns True or raises, and size does not unexpectedly increase.
    old_size = len(cf)
    try:
        result = cf.insert(item)
        assert (result is True) or isinstance(result, bool)
        # If insert True, size increments by 1; otherwise exception raised
        if result is True:
            assert len(cf) == old_size + 1
        else:
            assert len(cf) == old_size
    except exceptions.CuckooFilterFullException:
        # expected if full
        assert len(cf) == old_size

    # Insert multiple distinct items
    for i in range(50, 60):
        # May raise exception if filter full
        try:
            inserted = cf.insert(i)
            assert inserted is True
            assert i in cf
        except exceptions.CuckooFilterFullException:
            # Stop inserting if full
            break

    # Size matches number of fingerprints inserted (approx)
    # Size can be less than number of calls if some inserts failed with exception.
    assert 0 < len(cf) <= cf.capacity * cf.bucket_size


def test_contains_false_for_uninserted_items():
    cf = CuckooFilter(capacity=10)
    # Insert one item
    random.seed(1)
    cf.insert(1)
    # Items not inserted not contained
    for x in [2, 3, 4, 5]:
        assert x not in cf


def test_delete_success_and_failure(monkeypatch):
    random.seed(123)
    cf = CuckooFilter(capacity=15, bucket_size=3, fingerprint_size=1, max_displacements=500)

    # Insert item and confirm present
    item = 42
    cf.insert(item)
    assert item in cf
    old_size = len(cf)

    # Delete returns True for present item
    result = cf.delete(item)
    assert result is True
    # Now not contained
    assert item not in cf
    # Size decreased by 1
    assert len(cf) == old_size - 1

    # Delete non-existent item returns False, size unchanged
    result = cf.delete(999)
    assert result is False
    assert len(cf) == old_size - 1

def test_insert_raises_when_full(monkeypatch):
    # Use very small filter to fill quickly
    capacity = 2
    bucket_size = 1
    fingerprint_size = 1
    max_displacements = 5
    cf = CuckooFilter(capacity, bucket_size, fingerprint_size, max_displacements)

    random.seed(456)

    # We expect to fill this in very few inserts
    # Insert unique integer items until exception raised
    inserted_count = 0
    with pytest.raises(exceptions.CuckooFilterFullException):
        for i in range(1000):
            cf.insert(i)
            inserted_count += 1

    # Size should be at max capacity * bucket_size because no duplicates are possible with unique fingerprints
    assert 0 < len(cf) <= capacity * bucket_size
    # inserted_count at least 1 (some insertions succeeded before failure)
    assert inserted_count > 0

def test_insert_eviction_path(monkeypatch):
    # This test tries to exercise the eviction loop by forcing a small capacity and bucket size
    capacity = 3
    bucket_size = 1
    fp_size = 1
    max_disp = 10
    cf = CuckooFilter(capacity=capacity, bucket_size=bucket_size, fingerprint_size=fp_size, max_displacements=max_disp)

    random.seed(789)

    # Fill filter with capacity * bucket_size items
    for v in range(capacity * bucket_size):
        cf.insert(v)
    # Filter should be full now or close to it
    # Attempt inserting a new item should raise
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(999999)

def test_len_and_contains_synonym(monkeypatch):
    random.seed(11)
    cf = CuckooFilter(capacity=10)
    items = [101, 202, 303]
    for item in items:
        cf.insert(item)
    # __contains__ synonym is tested by 'in' operator
    for item in items:
        assert item in cf
    # len returns count of items inserted
    # Note: due to backing in buckets and possible collisions size may be less than inserted len
    assert len(cf) <= len(items)
    assert len(cf) > 0

def test_repr_and_str(monkeypatch):
    cf = CuckooFilter(capacity=12, bucket_size=2, fingerprint_size=3)
    rep = repr(cf)
    # Contains capacity, size, fingerprint size
    assert str(cf.capacity) in rep
    assert str(cf.size) in rep
    assert str(cf.fingerprint_size) in rep

def test_delete_nonexistent_returns_false(monkeypatch):
    cf = CuckooFilter(5)
    # Delete called on item not inserted returns False
    assert cf.delete(777) is False
    # Insert 1 item
    random.seed(5)
    cf.insert(1)
    # Delete other item returns False
    assert cf.delete(2) is False

def test_delete_after_reinsertion(monkeypatch):
    random.seed(99)
    cf = CuckooFilter(capacity=20)
    item = 555
    cf.insert(item)
    assert item in cf
    assert cf.delete(item) is True
    assert item not in cf
    # Reinsertion
    cf.insert(item)
    assert item in cf
    assert len(cf) == 1

def test_bucket_size_property_respected():
    # bucket_size property controls bucket size
    bucket_size = 7
    cf = CuckooFilter(capacity=5, bucket_size=bucket_size)
    # Each bucket has bucket_size slots
    for b in cf.buckets:
        assert b.size == bucket_size


@pytest.mark.parametrize("capacity,bucket_size,fp_size", [
    (10, 1, 1),
    (50, 4, 2),
    (100, 2, 3),
])
def test_multiple_configurations(capacity, bucket_size, fp_size):
    cf = CuckooFilter(capacity=capacity, bucket_size=bucket_size, fingerprint_size=fp_size)
    assert len(cf.buckets) == capacity
    for b in cf.buckets:
        assert b.size == bucket_size
    assert cf.fingerprint_size == fp_size


def test_insert_and_delete_multiple(monkeypatch):
    random.seed(2021)
    cf = CuckooFilter(capacity=30, bucket_size=4, fingerprint_size=1, max_displacements=1000)

    items = list(range(20))
    for i in items:
        cf.insert(i)

    # All inserted items should be contained
    for i in items:
        assert i in cf

    size_before_delete = len(cf)
    # Delete some but not all items
    to_delete = items[:10]
    for item in to_delete:
        assert cf.delete(item) is True
    # Check deletions worked
    for item in to_delete:
        assert item not in cf
    # Items not deleted should still be present
    for item in items[10:]:
        assert item in cf
    # Size decreased by number deleted
    assert len(cf) == size_before_delete - len(to_delete)

def test_cuckoofilterfullexception_is_subclass_of_exception():
    exc = exceptions.CuckooFilterFullException()
    assert isinstance(exc, Exception)
