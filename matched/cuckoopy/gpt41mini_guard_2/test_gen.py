import random
import pytest

from cuckoopy import cuckoofilter
from cuckoopy import exceptions


def test_init_attributes():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2, fingerprint_size=3, max_displacements=1000)
    assert cf.capacity == 10
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 3
    assert cf.max_displacements == 1000
    assert isinstance(cf.buckets, list)
    assert len(cf.buckets) == cf.capacity
    assert all(hasattr(b, 'insert') and hasattr(b, 'delete') and hasattr(b, 'swap') for b in cf.buckets)
    assert cf.size == 0


def test_repr_contains_len():
    cf = cuckoofilter.CuckooFilter(capacity=5)
    r = repr(cf)
    # Check repr format partially
    assert str(cf.capacity) in r
    assert str(cf.size) in r
    assert str(cf.fingerprint_size) in r
    # len returns size
    assert len(cf) == 0
    # contains uses contains method
    item = "test"
    # Initially no items present
    assert item not in cf


def test_insert_simple_and_contains():
    # To make test deterministic, seed random
    random.seed(42)
    cf = cuckoofilter.CuckooFilter(capacity=50, bucket_size=4, fingerprint_size=2, max_displacements=500)
    item = "apple"
    res = cf.insert(item)
    assert res is True
    # After insertion, size increased by 1
    assert len(cf) == 1
    # item should be contained
    assert item in cf
    # Insert the same item again implies another copy "stored"
    # Depending on bucket implementation, duplicates may be allowed
    res2 = cf.insert(item)
    assert res2 is True
    assert len(cf) == 2
    # Contains should still be True
    assert item in cf


def test_delete_existing_and_nonexisting_item():
    random.seed(123)
    cf = cuckoofilter.CuckooFilter(capacity=20, bucket_size=4, fingerprint_size=1, max_displacements=500)
    item = "banana"
    # Insert then delete
    inserted = cf.insert(item)
    assert inserted is True
    size_after_insert = len(cf)
    deleted = cf.delete(item)
    assert deleted is True
    # Size decreased by 1
    assert len(cf) == size_after_insert - 1
    # Item no longer contained
    assert item not in cf
    # Deleting again returns False (no such item)
    deleted_again = cf.delete(item)
    assert deleted_again is False


def test_contains_returns_false_for_uninserted_item():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert "non_existent_item" not in cf
    # Insert something else does not cause false positive here
    cf.insert("existing")
    assert "non_existent_item" not in cf


def test_insert_raises_when_full(monkeypatch):
    # To reliably test full condition: 
    # We'll create a small filter likely to fill quickly, 
    # and patch random.choice to always pick first bucket.
    cf = cuckoofilter.CuckooFilter(capacity=3, bucket_size=1, fingerprint_size=1, max_displacements=5)

    # Patch random.choice to always pick first argument (effectively deterministic eviction)
    monkeypatch.setattr(random, 'choice', lambda seq: seq[0])
    
    # Fill up the filter
    inserted = []
    for i in range(3):
        res = cf.insert(f'item-{i}')
        inserted.append(res)
        assert res is True
    # Next insert likely to fail (with above params)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('extra-item')


def test_insert_eviction_path(monkeypatch):
    # To test the eviction path deterministically, fix random.choice and fingerprints
    # Set max_displacements low to test the eviction loop runs but succeeds
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=1, fingerprint_size=1, max_displacements=10)

    monkeypatch.setattr(random, 'choice', lambda seq: seq[0])

    # Insert unique items to fill buckets partially
    items = [f'key{i}' for i in range(4)]
    # Insert normally
    for item in items:
        assert cf.insert(item) is True
    # Insert another item which triggers eviction path and eventually succeeds
    new_item = 'trigger-evict'
    assert cf.insert(new_item) is True
    # Size should be 5 after all insertions
    assert len(cf) == 5
    # Confirm those items exist logically (contains True)
    for item in items + [new_item]:
        assert item in cf


def test_delete_does_not_affect_other_fingerprints():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    items = ["a", "b", "c"]
    for item in items:
        assert cf.insert(item) is True
    size_before = len(cf)
    # Delete one item only
    deleted = cf.delete("b")
    assert deleted is True
    assert len(cf) == size_before - 1
    # Other items still contained
    assert "a" in cf
    assert "c" in cf
    # Deleted item no longer contained
    assert "b" not in cf


def test_insert_same_item_multiple_times_increments_size():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "dup-item"
    # Insert same item 3 times
    for _ in range(3):
        res = cf.insert(item)
        assert res is True
    # Size should reflect number of insertions (assuming duplicates allowed)
    assert len(cf) == 3
    # Deleting once should decrease size by 1
    assert cf.delete(item) is True
    assert len(cf) == 2


def test_contains_works_for_hash_collision_cases(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)

    # Patch hash_code to force collisions for test items but distinct fingerprints
    from cuckoopy import hashutils

    original_hash_code = hashutils.hash_code

    def fake_hash_code(item):
        # Return constant hash code to force collisions in index calculation
        return 42

    monkeypatch.setattr(hashutils, 'hash_code', fake_hash_code)

    # Insert two different items with same index but different fingerprints
    item1 = "collision1"
    item2 = "collision2"

    # To still have distinct fingerprints, patch fingerprint differently
    original_fingerprint = hashutils.fingerprint

    def fake_fingerprint(item, size):
        # Return bytes of different integer values based on item string for test
        return bytes([len(item)]) * size

    monkeypatch.setattr(hashutils, 'fingerprint', fake_fingerprint)

    # Insert item1
    assert cf.insert(item1) is True
    # Insert item2 (which has same bucket index but different fp)
    assert cf.insert(item2) is True

    # Both contained
    assert item1 in cf
    assert item2 in cf

    # Delete item1 only
    assert cf.delete(item1) is True
    # item2 should still be contained
    assert item2 in cf
    # item1 no longer contained
    assert item1 not in cf

    # Restore original functions - not strictly needed here as pytest cleans monkeypatch


def test_insert_returns_bool_and_size_is_correct(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2, fingerprint_size=1, max_displacements=100)

    # Patch random.choice to pick first to make deterministic eviction path
    monkeypatch.setattr(random, 'choice', lambda seq: seq[0])

    # Insert items and check inserted returns True
    for i in range(7):
        ret = cf.insert(f"item{i}")
        assert ret is True
        assert len(cf) == i + 1


def test_delete_non_existing_item_returns_false_and_size_unchanged():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    # Delete when empty returns False
    assert cf.delete("nope") is False
    initial_size = len(cf)
    cf.insert("exist")
    # Delete different item returns False
    assert cf.delete("nope") is False
    assert len(cf) == initial_size + 1


def test_contains_operator_and_contains_method_equivalent():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "x"
    # Initially not contained
    assert item not in cf
    assert cf.contains(item) is False
    # Inserted
    cf.insert(item)
    assert item in cf
    assert cf.contains(item) is True


def test_repr_contains_expected_info():
    cf = cuckoofilter.CuckooFilter(capacity=7, bucket_size=3, fingerprint_size=2)
    cf.insert("foo")
    r = repr(cf)
    assert f"capacity={cf.capacity}" in r
    assert f"size={cf.size}" in r
    assert f"fingerprint size={cf.fingerprint_size}" in r


def test_len_returns_size_and_updates_after_ops():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1
    cf.insert("b")
    assert len(cf) == 2
    cf.delete("a")
    assert len(cf) == 1
    cf.delete("nonexist")
    # size unchanged
    assert len(cf) == 1


def test_multiple_inserts_and_deletes_consistency(monkeypatch):
    random.seed(987)
    cf = cuckoofilter.CuckooFilter(capacity=15, bucket_size=4, fingerprint_size=1, max_displacements=50)

    items = [f'item{i}' for i in range(10)]
    # Insert all items
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 10

    # Delete half
    for item in items[:5]:
        assert cf.delete(item) is True
    assert len(cf) == 5
    # Confirm deleted not contained
    for item in items[:5]:
        assert item not in cf
    # Confirm others still contained
    for item in items[5:]:
        assert item in cf


def test_bucket_size_attribute_consistency():
    # Bucket size requested must be propagated to Bucket object (rough check)
    cf = cuckoofilter.CuckooFilter(capacity=6, bucket_size=7)
    assert cf.bucket_size == 7
    # All buckets have size 7
    assert all(hasattr(b, 'size') for b in cf.buckets)
    # bucket.Bucket.size attribute matches requested bucket_size
    sizes = [getattr(b, 'size', None) for b in cf.buckets]
    assert all(s == cf.bucket_size for s in sizes)

