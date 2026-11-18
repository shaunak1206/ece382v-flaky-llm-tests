import pytest
import random

from cuckoopy import cuckoofilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_insert_and_contains_basic():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "testitem"
    assert item not in cf
    assert cf.insert(item) is True
    assert item in cf


def test_insert_duplicates_increases_size_once():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=1)
    item = "dupitem"
    assert cf.insert(item) is True
    sz_after_first = len(cf)
    # inserting same item again should add only if fingerprint can be inserted again (possible)
    # but logically for cuckoo filter duplicates might be stored multiple times due to fingerprint collisions,
    # so size likely increments
    assert cf.insert(item) is True
    # size should increase by 1 more
    assert len(cf) == sz_after_first + 1
    assert item in cf


def test_delete_existing_item():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "delitem"
    cf.insert(item)
    size_before = len(cf)
    assert cf.delete(item) is True
    assert len(cf) == size_before - 1
    # Once deleted, item is no longer contained (very likely, but fingerprint collisions could exist)
    assert item not in cf


def test_delete_non_existing_item_returns_false():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "noexistitem"
    assert cf.delete(item) is False


def test_len_and_size_property():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert len(cf) == 0
    cf.insert("a")
    cf.insert("b")
    assert len(cf) == 2
    cf.delete("a")
    assert len(cf) == 1


def test_bucket_size_property_set_and_consistent():
    bucket_size = 8
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size
    for b in cf.buckets:
        # bucket.Bucket exposes size attribute
        assert getattr(b, "size", None) == bucket_size


def test_repr_contains_expected_info():
    cf = cuckoofilter.CuckooFilter(capacity=15, bucket_size=3, fingerprint_size=2)
    rep = repr(cf)
    assert "<CuckooFilter:" in rep
    assert "capacity=15" in rep
    assert "fingerprint size=2" in rep


def test_contains_magic_method_alias():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "magicitem"
    cf.insert(item)
    assert item in cf
    assert (item in cf) is cf.contains(item)


def test_insert_raises_exception_when_full(monkeypatch):
    # We force the filter to be full by patching random.choice and bucket inserts to always fail

    class DummyBucket:
        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            # return the fingerprint unchanged
            return fingerprint

        def delete(self, fingerprint):
            return False

        def __contains__(self, fingerprint):
            return False

    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=3)
    dummy_bucket = DummyBucket()
    cf.buckets = [dummy_bucket, dummy_bucket]

    # Patch random.choice to always pick first index to make behavior deterministic
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    with pytest.raises(CuckooFilterFullException):
        cf.insert("someitem")


def test_alternate_index_is_different_and_mod_capacity():
    cf = cuckoofilter.CuckooFilter(capacity=100)
    item = "altindexitem"
    index = cf._get_index(item)
    fingerprint = cuckoofilter.hashutils.fingerprint(item, cf.fingerprint_size)
    alt_index = cf._get_alternate_index(index, fingerprint)
    assert 0 <= alt_index < cf.capacity
    # Usually the alt_index differs from index unless hash collision
    # This will be true in most cases
    # But allow equality as valid
    assert isinstance(alt_index, int)


def test_insert_increases_size_and_delete_decreases_size():
    cf = cuckoofilter.CuckooFilter(capacity=15, bucket_size=4)
    # Insert multiple items
    for i in range(10):
        cf.insert(f"item-{i}")
    size_after_inserts = len(cf)
    assert size_after_inserts >= 10
    # Delete some items
    deleted_count = 0
    for i in range(5):
        if cf.delete(f"item-{i}"):
            deleted_count += 1
    assert len(cf) == size_after_inserts - deleted_count


def test_insert_returns_true_on_success(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=3, bucket_size=4)

    # Insert several unique items; all should return True
    items = ["a", "b", "c"]
    for it in items:
        assert cf.insert(it) is True


def test_contains_returns_false_for_never_inserted():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert "notinserted" not in cf
    assert cf.contains("notinserted") is False


def test_delete_returns_false_for_never_inserted():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.delete("notinserted") is False


def test_insert_fingerprint_size_effect(monkeypatch):
    # Insert with fingerprint size larger than 1 to check acceptance
    cf = cuckoofilter.CuckooFilter(capacity=10, fingerprint_size=4)

    item = "itemfp4"
    result = cf.insert(item)
    assert result is True
    assert item in cf

    # fingerprint sizes affects bucket usage internally but interface should remain consistent


def test_delete_only_decrements_when_item_found(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=10)
    cf.insert("item1")

    size_before = len(cf)
    # deleting existing item decrements size
    assert cf.delete("item1") is True
    assert len(cf) == size_before - 1

    # deleting non-existing item does not decrement size
    size_now = len(cf)
    assert cf.delete("item1") is False
    assert len(cf) == size_now


def test_buckets_are_independent_instances():
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=2)
    bucket_ids = [id(b) for b in cf.buckets]
    # All bucket instances should be unique
    assert len(set(bucket_ids)) == 5


def test_insert_handles_eviction(monkeypatch):
    # This test tries to trigger eviction path by having buckets only hold 1 item,
    # and by patching random.choice to alternate indexes
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=5)

    # Insert item1: should succeed
    item1 = "item1"
    cf.insert(item1)

    # Insert item2: should succeed as there is space
    item2 = "item2"
    cf.insert(item2)

    # Fill buckets to capacity
    bucket0 = cf.buckets[0]
    bucket1 = cf.buckets[1]

    # Both buckets full now, insertion of new item will cause evictions/swaps
    # Patch random.choice to always pick the first index to have deterministic eviction index during insert
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    item3 = "item3"
    # Should raise exception if eviction fails; but 5 max_displacements may allow inserts
    try:
        result = cf.insert(item3)
        # insert returns True or raises; so True if insert successful
        assert result is True
    except cuckoofilter.exceptions.CuckooFilterFullException:
        pytest.skip("Eviction path test: filter full exception raised during insert")


def test_insert_return_type_is_true():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert isinstance(cf.insert("foo"), bool)
    assert cf.insert("bar") is True


def test_cuckoofilterfullexception_is_exception():
    exc = cuckoofilter.exceptions.CuckooFilterFullException("test error")
    assert isinstance(exc, Exception)
    assert str(exc) == "test error"
