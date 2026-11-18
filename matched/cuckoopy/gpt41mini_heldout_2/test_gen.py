import pytest
import random

from cuckoopy import cuckoofilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_insert_and_contains_basic():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "test_item"
    assert cf.insert(item) is True
    assert item in cf
    assert cf.contains(item) is True
    assert len(cf) == 1


def test_insert_duplicate_item():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "duplicate_item"
    assert cf.insert(item) is True
    # Insert the same item again; filter allows duplicates:
    assert cf.insert(item) is True
    # Size increases by 2 as the insert is considered successful twice
    assert len(cf) == 2
    assert cf.contains(item) is True


def test_delete_existing_item():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "to_delete"
    assert cf.insert(item) is True
    assert cf.delete(item) is True
    assert cf.contains(item) is False
    assert len(cf) == 0


def test_delete_nonexistent_item():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "not_inserted"
    assert cf.delete(item) is False
    assert len(cf) == 0


def test_bucket_size_property():
    bucket_size = 7
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size


def test_repr_contains_expected_values():
    cf = cuckoofilter.CuckooFilter(capacity=5, fingerprint_size=2)
    rep = repr(cf)
    assert "CuckooFilter" in rep
    assert "capacity=5" in rep
    assert "fingerprint size=2 byte(s)" in rep
    assert f"size={len(cf)}" in rep


def test_len_and_size_consistency():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    assert len(cf) == 0
    cf.insert("a")
    cf.insert("b")
    assert len(cf) == 2
    cf.delete("a")
    assert len(cf) == 1


def test_insert_until_full_raises_exception(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1, max_displacements=3)

    # Patch random.choice to always pick the same bucket to make evictions predictable.
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    # First insert should succeed
    assert cf.insert("item1") is True

    # Next inserts will fail and eventually raise
    with pytest.raises(CuckooFilterFullException):
        cf.insert("item2")


def test_insert_eviction_success(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=10)

    # Insert an item filling the first bucket
    assert cf.insert("a") is True

    # Insert another item that will cause eviction:
    inserted_items = []

    # Patch random.choice to always pick bucket 0 for eviction to cause displacement
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    # Insert should succeed after evictions
    assert cf.insert("b") is True
    assert len(cf) == 2

    # Both items should be contained
    assert "a" in cf
    assert "b" in cf


def test_contains_false_for_missing_item():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    cf.insert("existing")
    assert not cf.contains("missing")
    assert "missing" not in cf


def test_delete_decreases_size_only_once_for_duplicates():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "dup_item"
    cf.insert(item)
    cf.insert(item)
    assert len(cf) == 2
    assert cf.delete(item) is True
    # Only one deleted, size reduces by one
    assert len(cf) == 1
    # Deleting again removes the second
    assert cf.delete(item) is True
    assert len(cf) == 0
    # Further deleting returns False
    assert cf.delete(item) is False


def test_get_index_and_get_alternate_index_consistency():
    cf = cuckoofilter.CuckooFilter(capacity=100, bucket_size=4)
    item = "some_item"
    idx = cf._get_index(item)
    fp = cf.fingerprint_size
    fingerprint = cf.fingerprint_size == 1 and list(cf.buckets[0]) or None
    # Just test _get_alternate_index returns an int within valid range
    alt_idx = cf._get_alternate_index(idx,
                                      cuckoofilter.hashutils.fingerprint(item, cf.fingerprint_size))
    assert isinstance(idx, int)
    assert 0 <= idx < cf.capacity
    assert isinstance(alt_idx, int)
    assert 0 <= alt_idx < cf.capacity


def test_insert_raises_when_bucket_swap_fails(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1, max_displacements=1)

    # Patch buckets to simulate swap always returns same fingerprint without freeing space
    class FakeBucket:
        def insert(self, fp):
            return False

        def swap(self, fp):
            return fp

        def delete(self, fp):
            return False

        def __contains__(self, fp):
            return False

    cf.buckets = [FakeBucket()]
    with pytest.raises(CuckooFilterFullException):
        cf.insert("fail_swap")


def test_insert_increases_size_correctly(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)

    item1 = "item1"
    item2 = "item2"
    initial_size = len(cf)

    assert cf.insert(item1) is True
    assert len(cf) == initial_size + 1

    assert cf.insert(item2) is True
    assert len(cf) == initial_size + 2


def test_delete_decreases_size_correctly():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)

    item1 = "item1"
    item2 = "item2"

    cf.insert(item1)
    cf.insert(item2)
    before_delete_size = len(cf)

    assert cf.delete(item1) is True
    assert len(cf) == before_delete_size - 1
    assert cf.delete(item2) is True
    assert len(cf) == before_delete_size - 2


def test_contains_is_magic_method():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)

    item = "magic_in"
    cf.insert(item)
    assert (item in cf) is True

    nonexisting = "magic_not_in"
    assert (nonexisting in cf) is False


def test_size_not_negative_on_multiple_deletes():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4)
    item = "item"
    cf.insert(item)
    cf.delete(item)
    # delete again should return False but not reduce size below zero
    assert cf.delete(item) is False
    assert len(cf) == 0
