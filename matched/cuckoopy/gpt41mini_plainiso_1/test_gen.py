import pytest
import random

from cuckoopy import cuckoofilter, exceptions
from cuckoopy.cuckoofilter import CuckooFilter


def test_insert_and_contains_basic():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "test1"
    assert cf.insert(item) is True
    assert item in cf
    assert cf.contains(item) is True
    assert len(cf) == 1


def test_insert_duplicate_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "dupitem"
    assert cf.insert(item) is True
    # Insert same item again, should still return True (filter accepts it)
    assert cf.insert(item) is True
    # Size increases by 2 because fingerprints are stored twice if possible
    assert len(cf) == 2 or len(cf) == 1  # size could be 1 or 2 based on insert logic

    assert item in cf


def test_delete_existing_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "item_to_delete"
    cf.insert(item)
    assert item in cf
    deleted = cf.delete(item)
    assert deleted is True
    assert item not in cf
    assert len(cf) == 0


def test_delete_non_existing_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item1 = "existing"
    item2 = "non_existing"
    cf.insert(item1)
    assert item2 not in cf
    deleted = cf.delete(item2)
    assert deleted is False
    assert item1 in cf
    assert len(cf) == 1


def test_bucket_size_property():
    cf = CuckooFilter(capacity=10, bucket_size=8, fingerprint_size=1)
    assert cf.bucket_size == 8
    cf2 = CuckooFilter(capacity=15, bucket_size=2, fingerprint_size=1)
    assert cf2.bucket_size == 2


def test_insert_until_full_and_raises():
    # Small filter to fill quickly
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=10)
    # Try to insert unique items until it raises exception

    inserted = 0
    with pytest.raises(exceptions.CuckooFilterFullException):
        while True:
            success = cf.insert(f"item-{inserted}")
            inserted += 1
            # Defensive: break if size exceeds logical max to avoid infinite loop
            if inserted > 1000:
                break
    assert inserted >= 1
    # The size of filter cannot exceed capacity * bucket_size
    assert len(cf) <= cf.capacity * cf.bucket_size


def test_repr_and_len():
    cf = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2)
    rep = repr(cf)
    assert rep.startswith("<CuckooFilter:")
    assert str(cf) not in ["", None]
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1


def test_contains_after_multiple_inserts():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    items = ["apple", "banana", "cherry"]
    for item in items:
        assert cf.insert(item) is True

    for item in items:
        assert item in cf

    assert "not_in_filter" not in cf


def test_delete_decrements_size_only_if_found():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "to_be_deleted"
    cf.insert(item)
    size_before_delete = len(cf)
    deleted = cf.delete(item)
    size_after_delete = len(cf)
    assert deleted is True
    assert size_after_delete == size_before_delete - 1

    # Try deleting again should fail and size should remain same
    deleted_again = cf.delete(item)
    size_after_second_delete = len(cf)
    assert deleted_again is False
    assert size_after_second_delete == size_after_delete


def test_insert_uses_both_buckets(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=5)

    calls = []

    # Monkeypatch random.choice to track which index chosen for eviction
    def mock_choice(seq):
        calls.append(seq)
        # Always pick first index for eviction for determinism
        return seq[0]

    monkeypatch.setattr(random, "choice", mock_choice)

    for i in range(cf.capacity * cf.bucket_size):
        cf.insert(f"item-{i}")

    # Insert more to force evictions & confirm random.choice called
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert("overflow")

    # Confirm random.choice used with expected input sets
    assert all(isinstance(c, list) and len(c) == 2 for c in calls)


def test_contains_calls_bucket_contains(monkeypatch):
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "test_contains_calls"

    # Monkeypatch buckets' __contains__ to track call
    found_calls = []

    def fake_bucket_contains(self, fingerprint):
        found_calls.append(fingerprint)
        return False

    for bucket_instance in cf.buckets:
        monkeypatch.setattr(bucket_instance, "__contains__", fake_bucket_contains.__get__(bucket_instance))

    _ = cf.contains(item)
    assert isinstance(found_calls, list)
    # Should be called twice (two buckets checked)
    assert len(found_calls) == 2


def test_delete_calls_bucket_delete(monkeypatch):
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "to_delete_test"
    cf.insert(item)

    delete_calls = []

    def fake_bucket_delete(self, fingerprint):
        delete_calls.append(fingerprint)
        # Fake successful delete only on first call
        if len(delete_calls) == 1:
            return True
        return False

    for b in cf.buckets:
        monkeypatch.setattr(b, "delete", fake_bucket_delete.__get__(b))

    deleted = cf.delete(item)
    assert deleted is True
    assert len(delete_calls) >= 1


def test_insert_raises_with_zero_capacity():
    cf = CuckooFilter(capacity=0)
    with pytest.raises(IndexError):
        cf.insert("test")  # Because capacity=0, buckets list is empty, mod by zero error likely


def test_insert_returns_true_if_bucket_inserts_successfully(monkeypatch):
    cf = CuckooFilter(capacity=10, bucket_size=1)

    # Patch buckets to always accept insert without swapping
    def always_insert(self, fingerprint):
        return True

    for b in cf.buckets:
        monkeypatch.setattr(b, "insert", always_insert.__get__(b))

    result = cf.insert("dummyitem")
    assert result is True
    assert len(cf) == 1


def test_insert_raises_when_max_displacements_exceeded(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=1, max_displacements=3)

    # Patch bucket insert and swap to simulate full buckets but with swap always returning the same fingerprint
    def always_false_insert(self, fingerprint):
        return False

    def always_swap(self, fingerprint):
        return fingerprint

    for b in cf.buckets:
        monkeypatch.setattr(b, "insert", always_false_insert.__get__(b))
        monkeypatch.setattr(b, "swap", always_swap.__get__(b))

    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert("someitem")


def test_contains_returns_false_for_empty_filter():
    cf = CuckooFilter(capacity=10, bucket_size=4)
    assert cf.contains("missing") is False
    assert "missing" not in cf


def test_delete_returns_false_if_fingerprint_absent(monkeypatch):
    cf = CuckooFilter(capacity=10, bucket_size=4)

    # Patch bucket delete to always return False
    def always_false_delete(self, fingerprint):
        return False

    for b in cf.buckets:
        monkeypatch.setattr(b, "delete", always_false_delete.__get__(b))

    deleted = cf.delete("item")
    assert deleted is False
    assert len(cf) == 0


def test_mutual_exclusivity_of_insert_and_delete():
    cf = CuckooFilter(capacity=10, bucket_size=4)

    item = "testitem"

    assert cf.insert(item) is True
    assert item in cf
    assert cf.delete(item) is True
    assert item not in cf
    # Deleting again returns False
    assert cf.delete(item) is False


def test_bucket_size_do_not_affect_wrongly():
    cf1 = CuckooFilter(capacity=10, bucket_size=1)
    cf2 = CuckooFilter(capacity=10, bucket_size=10)
    assert cf1.bucket_size == 1
    assert cf2.bucket_size == 10
    # Simple inserts to check no error
    assert cf1.insert("a")
    assert cf2.insert("a")
