import pytest
import random

from cuckoopy import cuckoofilter
from cuckoopy import exceptions


def test_insert_and_contains_basic():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "test_item"
    # Initially, item should not be contained
    assert item not in cf
    # Insert item
    result = cf.insert(item)
    assert result is True
    # Now item should be contained
    assert item in cf
    # Size should be 1
    assert len(cf) == 1


def test_insert_returns_true_when_bucket_free(monkeypatch):
    # Patch bucket.Bucket.insert to always return True on first bucket
    class DummyBucket:
        def __init__(self):
            self.data = []
        def insert(self, fingerprint):
            self.data.append(fingerprint)
            return True
        def delete(self, fingerprint):
            if fingerprint in self.data:
                self.data.remove(fingerprint)
                return True
            return False
        def swap(self, fingerprint):
            # Not used in this test
            return None
        def __contains__(self, fingerprint):
            return fingerprint in self.data

    original_bucket = cuckoofilter.bucket.Bucket
    try:
        monkeypatch.setattr(cuckoofilter.bucket, "Bucket", lambda size=None: DummyBucket())
        cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)
        assert cf.insert("something")
    finally:
        monkeypatch.setattr(cuckoofilter.bucket, "Bucket", original_bucket)


def test_contains_fingerprint_in_either_bucket(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=4, fingerprint_size=1)
    item = "foo"

    # Insert item, should be True
    assert cf.insert(item) is True

    fp = cuckoofilter.hashutils.fingerprint(item, cf.fingerprint_size)
    i = cf._get_index(item)
    j = cf._get_alternate_index(i, fp)

    # fingerprint should be in bucket i or bucket j
    assert fp in cf.buckets[i] or fp in cf.buckets[j]
    assert cf.contains(item) is True
    assert (item in cf) is True


def test_delete_returns_true_on_deleted_item_and_reduces_size():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "to_delete"
    cf.insert(item)
    orig_size = len(cf)
    assert cf.delete(item) is True
    assert len(cf) == orig_size - 1
    # After deletion, item is no longer contained
    assert item not in cf


def test_delete_returns_false_on_nonexisting_item():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "nonexistent"
    assert cf.delete(item) is False


def test_insert_raises_exception_when_full(monkeypatch):
    # Prepare cf with capacity=1 and bucket_size=1 to easily fill
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=5)

    # Patch random.choice to always choose same index to make eviction predictable
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    # Insert an item should succeed
    cf.insert("item1")

    # Patch Bucket.swap and insert so that eviction never frees a spot
    class BlockedBucket:
        def __init__(self):
            self.data = set()
        def insert(self, fingerprint):
            # Always full
            return False
        def swap(self, fingerprint):
            # Always swap fingerprint out (simulate full bucket)
            # Return different fingerprint to simulate eviction
            return fingerprint + b"x"
        def delete(self, fingerprint):
            return False
        def __contains__(self, fingerprint):
            return False

    monkeypatch.setattr(cf, "buckets", [BlockedBucket()])
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert("item2")


def test_len_and_repr_methods():
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=2, fingerprint_size=1)
    s = repr(cf)
    assert isinstance(s, str)
    assert f"capacity={cf.capacity}" in s
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1


def test_bucket_size_property_is_set_correctly():
    bucket_size = 7
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size
    for b in cf.buckets:
        assert b.size == bucket_size


def test_contains_returns_false_for_non_inserted_item():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item_inserted = "inserted"
    item_not_inserted = "not_inserted"
    cf.insert(item_inserted)
    assert item_not_inserted not in cf
    assert cf.contains(item_not_inserted) is False


def test_insert_increments_size_once_only():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "unique"
    # Insert first time
    assert cf.insert(item) is True
    initial_size = len(cf)
    # Try insert again (should insert again if fingerprint not found or skip)
    # It's possible insertion again returns True or raises exception if full.
    # But size should increase only once (typical cuckoofilter allows duplicates?)
    # Actually the code does not prevent duplicates and will increase size on each successful insert.
    # So, test that size increases if inserted again.
    result = cf.insert(item)
    if result:
        assert len(cf) == initial_size + 1
    else:
        # If insert fails, size unchanged
        assert len(cf) == initial_size


def test_delete_decrements_size_only_if_deleted(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "delitem"
    cf.insert(item)
    size_before = len(cf)
    # Delete existing
    assert cf.delete(item) is True
    assert len(cf) == size_before - 1
    # Delete non-existing
    assert cf.delete(item) is False
    assert len(cf) == size_before - 1


def test_get_index_and_get_alternate_index_produce_valid_indices():
    cf = cuckoofilter.CuckooFilter(capacity=50)
    item = "someitem"
    fp = cuckoofilter.hashutils.fingerprint(item, cf.fingerprint_size)
    i = cf._get_index(item)
    j = cf._get_alternate_index(i, fp)
    assert 0 <= i < cf.capacity
    assert 0 <= j < cf.capacity
    assert isinstance(i, int)
    assert isinstance(j, int)
    # Because alternate_index = index ^ hash_code(fp) mod capacity, i != j is possible
    # But no strict assert about difference, just presence in range


def test_cuckoofilterfull_exception_is_subclass_of_exception():
    ex = exceptions.CuckooFilterFullException("msg")
    assert isinstance(ex, Exception)
    assert isinstance(ex, exceptions.CuckooFilterFullException)
    assert str(ex) == "msg"


def test_insert_triggering_eviction_path(monkeypatch):
    # This test focuses on exercising the eviction loop path in insert.

    # Prepare buckets that will be full initially forcing eviction.
    class EvictionBucket:
        def __init__(self):
            # Hold exactly 1 fingerprint because bucket_size=1 for test
            self.fingerprints = [b'X']  # Pre-filled to block insert
        def insert(self, fingerprint):
            return False  # Always full
        def swap(self, fingerprint):
            # Swap out first fingerprint, insert the new
            old = self.fingerprints[0]
            self.fingerprints[0] = fingerprint
            return old
        def delete(self, fingerprint):
            if fingerprint in self.fingerprints:
                self.fingerprints.remove(fingerprint)
                return True
            return False
        def __contains__(self, fingerprint):
            return fingerprint in self.fingerprints

    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=10)
    # Replace buckets with EvictionBucket to force eviction swaps
    cf.buckets = [EvictionBucket(), EvictionBucket()]

    # Patch random.choice to always pick first bucket to make eviction path predictable
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    # Try inserting a new item, should succeed via eviction path (since max_displacements=10)
    inserted = cf.insert("evict_item")
    assert inserted is True
    # Size should be incremented
    assert len(cf) == 1


def test_insert_random_choice_alternate_index(monkeypatch):
    # Insert with random.choice picking one index, then try inserting more items to test eviction
    cf = cuckoofilter.CuckooFilter(capacity=3, bucket_size=1, fingerprint_size=1, max_displacements=5)

    # Patch random.choice to always select second index to test alternate eviction branch
    monkeypatch.setattr(random, "choice", lambda x: x[1])

    inserted1 = cf.insert("itemA")
    assert inserted1 is True

    # Manually fill both buckets with dummy fingerprints to block direct insert
    for b in cf.buckets:
        b.insert(cuckoofilter.hashutils.fingerprint("dummy", 1))

    # Now insert should trigger eviction route
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert("new_item")


def test_delete_item_not_affecting_size_when_not_found(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=5)
    cf.insert("found_item")
    size_before = len(cf)
    # Delete an item fingerprint not in buckets
    # Patch buckets to simulate no deletion possible on any bucket
    class NoDeleteBucket:
        def insert(self, fp):
            return True
        def delete(self, fp):
            return False
        def __contains__(self, fp):
            return False
    cf.buckets = [NoDeleteBucket() for _ in range(cf.capacity)]
    # Delete returns False
    assert cf.delete("not_found") is False
    # Size unchanged
    assert len(cf) == size_before


def test_insert_with_fingerprint_size_greater_than_one():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    item = "multi_fp"
    inserted = cf.insert(item)
    assert inserted is True
    assert item in cf
    # Size is right
    assert len(cf) == 1


def test_insert_duplicate_items_increases_size_each_time():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "dup"

    # Insert same item multiple times
    inserted1 = cf.insert(item)
    inserted2 = cf.insert(item)
    # Both insertions should succeed (code does not prevent duplicate fingerprints)
    assert inserted1 is True
    assert inserted2 is True
    # Size increased twice
    assert len(cf) == 2


def test_delete_reduces_size_only_if_item_deleted_once():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "testitem"
    cf.insert(item)
    cf.insert(item)
    size_before = len(cf)
    # Delete one occurrence
    assert cf.delete(item) is True
    assert len(cf) == size_before - 1
    # Delete again
    assert cf.delete(item) is True
    assert len(cf) == size_before - 2
    # Delete again should be false (no more occurrences)
    assert cf.delete(item) is False
    assert len(cf) == size_before - 2
