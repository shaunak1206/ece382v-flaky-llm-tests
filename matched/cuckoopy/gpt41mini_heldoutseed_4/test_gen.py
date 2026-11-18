import pytest
import random

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_insert_and_contains():
    cf = CuckooFilter(capacity=100, bucket_size=4, fingerprint_size=1)
    item = "testitem"
    assert cf.insert(item) is True
    assert item in cf
    assert cf.contains(item) is True


def test_insert_multiple_items():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    items = [f"item{i}" for i in range(5)]
    for it in items:
        assert cf.insert(it) is True
    for it in items:
        assert cf.contains(it) is True
    assert len(cf) == 5


def test_delete_existing_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "to_delete"
    cf.insert(item)
    assert cf.delete(item) is True
    assert item not in cf
    assert cf.contains(item) is False
    assert len(cf) == 0


def test_delete_nonexistent_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "not_inserted"
    assert cf.delete(item) is False


def test_bucket_size_property():
    bucket_size = 5
    cf = CuckooFilter(capacity=10, bucket_size=bucket_size, fingerprint_size=1)
    assert cf.bucket_size == bucket_size


def test_insert_until_full_then_raises(monkeypatch):
    # Patch random.choice to always pick the same bucket to cause predictable eviction cycle.
    # Also patch Bucket.swap and Bucket.insert to simulate evictions failing after max_displacements.

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=3)

    # Insert first 2 items to fill both buckets.
    inserted_items = ["itemA", "itemB"]
    for item in inserted_items:
        assert cf.insert(item) is True

    # All buckets will be full now, next insert should fail.

    # Patch random.choice to always pick the first index to cause deterministic eviction
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    # Patch swap method to always return the fingerprint passed in, so insertion never frees a spot.
    # Also patch insert method to always return False after full.
    for b in cf.buckets:
        monkeypatch.setattr(b, "swap", lambda fingerprint: fingerprint)
        # ensure insert returns False for eviction cause to continue
        monkeypatch.setattr(b, "insert", lambda fingerprint: False)

    with pytest.raises(CuckooFilterFullException):
        cf.insert("new_item")


def test_len_and_size_consistency():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    assert len(cf) == 0
    cf.insert("a")
    cf.insert("b")
    assert len(cf) == 2
    cf.delete("a")
    assert len(cf) == 1


def test_repr_contains_correct_info():
    cf = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2)
    r = repr(cf)
    assert "capacity=5" in r
    assert "size=0" in r
    assert "fingerprint size=2 byte(s)" in r
    cf.insert("item")
    r2 = repr(cf)
    assert "size=1" in r2


def test_contains_returns_false_for_absent_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    # Insert a different item
    cf.insert("present")
    assert cf.contains("absent") is False
    assert "absent" not in cf


def test_insert_adds_size_only_once_even_if_both_buckets_accept(monkeypatch):
    # We want to simulate that both buckets accept fingerprint,
    # but insert method should count size only once.

    cf = CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)

    # Patch the two buckets insert to always return True
    # But we reset size to 0 before

    insert_call_counts = []

    def insert_and_record(fingerprint):
        insert_call_counts.append(True)
        return True

    monkeypatch.setattr(cf.buckets[0], "insert", insert_and_record)
    monkeypatch.setattr(cf.buckets[1], "insert", insert_and_record)

    res = cf.insert("dup_item")

    # The insert method attempts i, then j, but since first returns True, second won't be called,
    # so only one call actually happens in real flow. Here both buckets' insert replaced with same counting fn,
    # but insert returns True on first bucket insert, so second bucket insert is never called.

    assert res is True
    assert len(insert_call_counts) == 1
    assert len(cf) == 1


def test_delete_decrements_size_only_when_deletion_happens(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)

    # Insert one item
    cf.insert("item")

    # Patch buckets to simulate deleting fingerprint only on first bucket
    deleted_flags = []

    def delete_true(finger):
        deleted_flags.append(True)
        return True

    def delete_false(finger):
        deleted_flags.append(False)
        return False

    monkeypatch.setattr(cf.buckets[0], "delete", delete_true)
    monkeypatch.setattr(cf.buckets[1], "delete", delete_false)

    # Deleting should reduce size by 1
    result = cf.delete("item")
    assert result is True
    assert len(cf) == 0

    # Now patch both buckets delete to always False i.e. nothing deleted
    monkeypatch.setattr(cf.buckets[0], "delete", delete_false)
    monkeypatch.setattr(cf.buckets[1], "delete", delete_false)

    result2 = cf.delete("item")
    assert result2 is False
    # Size remains zero
    assert len(cf) == 0


def test_inserting_same_item_twice_increases_size_twice(monkeypatch):
    # Note: Because of fingerprint collisions, inserting same item twice inserts two entries.
    cf = CuckooFilter(capacity=5, bucket_size=4, fingerprint_size=1)

    cf.insert("duplicate")
    length_after_first = len(cf)
    cf.insert("duplicate")
    length_after_second = len(cf)

    # Length should have increased by 1 for each insert, since filter doesn't prevent duplicates
    assert length_after_second == length_after_first + 1


def test_bucket_insert_called_with_correct_fingerprint(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)

    recorded_fingerprints = []

    original_insert = cf.buckets[0].insert

    def fake_insert(fingerprint):
        recorded_fingerprints.append(fingerprint)
        return original_insert(fingerprint)

    monkeypatch.setattr(cf.buckets[0], "insert", fake_insert)

    item = "some_item"
    cf.insert(item)

    # The fingerprint passed to insert should be a bytes object (fingerprint size=1 byte)
    for fp in recorded_fingerprints:
        assert isinstance(fp, bytes)
        assert len(fp) == 1


def test_contains_when_both_buckets_empty(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)

    # Patch both buckets __contains__ to always False
    monkeypatch.setattr(cf.buckets[0], "__contains__", lambda self, item: False)
    monkeypatch.setattr(cf.buckets[1], "__contains__", lambda self, item: False)

    assert not cf.contains("missing_item")
    assert "missing_item" not in cf


def test_delete_with_fingerprint_collision(monkeypatch):
    # We test that delete only removes fingerprint if exact match found.

    cf = CuckooFilter(capacity=2, bucket_size=4, fingerprint_size=1)
    item1 = "item1"
    item2 = "item2"

    # Insert item1
    cf.insert(item1)
    length_before = len(cf)

    # Patch buckets delete to simulate false negative for item2 (simulate fingerprint collision)
    def delete_stub(fingerprint):
        # Simulate never deleting
        return False

    monkeypatch.setattr(cf.buckets[cf._get_index(item2)], "delete", delete_stub)
    monkeypatch.setattr(cf.buckets[cf._get_alternate_index(cf._get_index(item2),
                                                          cf.buckets[0].insert), "delete"], delete_stub)

    # Deleting item2 (not inserted) returns False and no size decrement
    result = cf.delete(item2)
    assert result is False
    assert len(cf) == length_before


def test_insert_with_zero_max_displacements_raises():
    # max_displacements=0 means no evictions allowed, so once buckets full insert fails
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    cf.insert("item1")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("item2")
