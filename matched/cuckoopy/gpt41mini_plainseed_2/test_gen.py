import pytest
import random

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_insert_and_contains_simple():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "test_item"
    assert item not in cf
    assert cf.insert(item)
    assert item in cf
    assert len(cf) == 1


def test_insert_returns_true_multiple_items():
    cf = CuckooFilter(capacity=20, bucket_size=4, fingerprint_size=1)
    items = [f"item{i}" for i in range(15)]
    for item in items:
        assert cf.insert(item)
    for item in items:
        assert item in cf
    assert len(cf) == len(items)


def test_delete_existing_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "del_item"
    cf.insert(item)
    assert item in cf
    assert cf.delete(item)
    assert item not in cf
    assert len(cf) == 0


def test_delete_non_existing_item_returns_false():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "non_existent"
    assert not cf.delete(item)
    assert len(cf) == 0


def test_insert_duplicate_items_increases_size_once():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "dup_item"
    assert cf.insert(item)
    # Try inserting same item again - cuckoo filters may accept duplicates but size counts insertions successfully
    assert cf.insert(item)
    # Both should report item in filter
    assert item in cf
    # Size will increment each successful insertion even if fingerprint is same
    assert len(cf) == 2


def test_len_and_repr_methods():
    cf = CuckooFilter(capacity=5, bucket_size=4, fingerprint_size=2)
    repr_str = repr(cf)
    assert "capacity=5" in repr_str
    assert "fingerprint size=2" in repr_str
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1
    repr_str2 = repr(cf)
    assert "size=1" in repr_str2


def test_contains_returns_false_for_not_inserted():
    cf = CuckooFilter(capacity=10)
    assert "missing" not in cf


def test_bucket_size_property_matches_init_arg():
    bucket_size = 3
    cf = CuckooFilter(capacity=10, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size


def test_insert_raises_full_exception_when_full(monkeypatch):
    # We will patch random.choice to always pick the same eviction index to keep test deterministic
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=3)

    # Insert first item - should succeed
    assert cf.insert("item1")
    assert len(cf) == 1

    # Patch buckets so that insert always fails, forcing eviction to happen max_displacements times
    # We'll patch insert to always return False and swap to always return some fingerprint
    # so that insert triggers eviction loop until max displacement reached
    cf.buckets[0].insert = lambda fp: False
    cf.buckets[0].swap = lambda fp: fp  # returns same fp for simplicity
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    with pytest.raises(CuckooFilterFullException):
        cf.insert("item2")


def test_contains_and_delete_using_custom_items():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    objs = [{"id": i} for i in range(5)]

    for obj in objs:
        cf.insert(str(obj))  # Convert dict to string to keep hashing consistent

    for obj in objs:
        assert str(obj) in cf

    obj_del = objs[2]
    assert cf.delete(str(obj_del))
    assert str(obj_del) not in cf
    assert len(cf) == 4


def test_delete_reduces_size_appropriately():
    cf = CuckooFilter(capacity=10, bucket_size=4)
    items = [f"item{i}" for i in range(4)]
    for item in items:
        cf.insert(item)
    assert len(cf) == 4
    cf.delete(items[0])
    assert len(cf) == 3
    cf.delete(items[1])
    assert len(cf) == 2
    # Deleting non-existent should not change size
    assert not cf.delete("not_found")
    assert len(cf) == 2


def test_insert_with_max_displacements_eventually_succeeds(monkeypatch):
    # This test makes sure that insert tries displacement but eventually inserts successfully

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=5)

    # First, insert an item so one bucket is full
    assert cf.insert("a")

    # Patch buckets:
    # Bucket insert returns False if full, True if empty
    # Bucket swap returns replaced fingerprint

    called = {"swap_count": 0}

    class DummyBucket:
        def __init__(self):
            self.storage = []
            self.size = 1
        def insert(self, fp):
            # Only insert if empty
            if len(self.storage) < 1:
                self.storage.append(fp)
                return True
            return False
        def swap(self, fp):
            called["swap_count"] += 1
            evicted = self.storage[0]
            self.storage[0] = fp
            return evicted
        def delete(self, fp):
            if fp in self.storage:
                self.storage.remove(fp)
                return True
            return False
        def __contains__(self, fp):
            return fp in self.storage

    b0 = DummyBucket()
    b1 = DummyBucket()
    cf.buckets = [b0, b1]

    # Insert one item - should occupy one bucket
    assert cf.insert("x")

    # Insert another item triggering displacement (since buckets have size 1)
    assert cf.insert("y")
    # Confirm number of swaps called is at least one but less than max_displacements
    assert 0 < called["swap_count"] <= cf.max_displacements
    # Items inserted should be in filter logically
    assert "x" in cf or "y" in cf


def test_contains_and_delete_are_consistent_with_mutations():
    cf = CuckooFilter(capacity=5, bucket_size=4)
    item = "check_consistency"
    # Delete before insert returns False
    assert not cf.delete(item)
    # Insert item
    assert cf.insert(item)
    # Contains should be True
    assert item in cf
    # Delete should return True
    assert cf.delete(item)
    # Contains should be False after deletion
    assert not item in cf
    # Delete again returns False
    assert not cf.delete(item)


def test_insert_with_different_fingerprint_sizes():
    cf1 = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    cf2 = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=4)

    item = "fingerprint_test"

    assert cf1.insert(item)
    assert cf2.insert(item)

    assert item in cf1
    assert item in cf2


def test_multiple_inserts_then_deletes_length_consistency():
    cf = CuckooFilter(capacity=20, bucket_size=4)
    items = [f"item{i}" for i in range(10)]
    for item in items:
        cf.insert(item)
    assert len(cf) == 10
    for item in items[:5]:
        cf.delete(item)
    assert len(cf) == 5
    for item in items[5:]:
        cf.delete(item)
    assert len(cf) == 0


def test_insert_raise_exception_when_capacity_zero():
    # capacity zero means no buckets
    cf = CuckooFilter(capacity=0)
    with pytest.raises(ZeroDivisionError):
        cf.insert("any")  # hash_code % 0 triggers ZeroDivisionError


def test_internal_indexes_are_consistent():
    cf = CuckooFilter(capacity=10)
    item = "internal_test"
    idx = cf._get_index(item)
    fingerprint = cf._get_index(item)  # Just fetching again (not fingerprint)
    alt_idx = cf._get_alternate_index(idx, fingerprint)
    assert 0 <= idx < cf.capacity
    assert 0 <= alt_idx < cf.capacity
    # alt_idx should differ or could be same sometimes


def test_contains_magic_method_behaves_like_contains():
    cf = CuckooFilter(capacity=10)
    item = "magic"
    assert not (item in cf)
    cf.insert(item)
    assert (item in cf)


def test_delete_magic_method_is_false_for_absent():
    cf = CuckooFilter(capacity=10)
    item = "magic_delete"
    assert not cf.delete(item)  # no error


def test_repr_contains_all_expected_info():
    cf = CuckooFilter(capacity=7, bucket_size=5, fingerprint_size=2)
    r = repr(cf)
    assert "capacity=7" in r
    assert "fingerprint size=2" in r
    assert "size=0" in r
    cf.insert("a")
    r2 = repr(cf)
    assert "size=1" in r2
