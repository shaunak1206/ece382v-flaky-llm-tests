import pytest
import random

import cuckoopy.cuckoofilter as cf
import cuckoopy.exceptions as exceptions
from cuckoopy import bucket, hashutils


def test_insert_single_item():
    f = cf.CuckooFilter(capacity=10)
    item = "test_item"
    assert f.insert(item) is True
    assert len(f) == 1
    assert item in f


def test_contains_returns_true_for_inserted_item():
    f = cf.CuckooFilter(capacity=20)
    item = "hello"
    f.insert(item)
    assert f.contains(item) is True
    assert item in f


def test_contains_returns_false_for_noninserted_item():
    f = cf.CuckooFilter(capacity=20)
    f.insert("abc")
    assert f.contains("notinserted") is False


def test_delete_existing_item():
    f = cf.CuckooFilter(capacity=15)
    item = "delete_me"
    f.insert(item)
    assert f.delete(item) is True
    assert len(f) == 0
    assert f.contains(item) is False
    assert item not in f


def test_delete_nonexistent_item():
    f = cf.CuckooFilter(capacity=15)
    f.insert("some_item")
    assert f.delete("absent_item") is False
    assert len(f) == 1


def test_bucket_size_property():
    bucket_size = 7
    f = cf.CuckooFilter(capacity=5, bucket_size=bucket_size)
    assert f.bucket_size == bucket_size
    for b in f.buckets:
        assert isinstance(b, bucket.Bucket)
        assert b.size == bucket_size


def test_insert_raises_when_full(monkeypatch):
    """
    Fill the filter forcibly so that insert does not find space after max_displacements
    """

    class DummyBucket:
        def __init__(self):
            self.calls = 0

        def insert(self, fingerprint):
            return False  # Never insert

        def swap(self, fingerprint):
            self.calls += 1
            return fingerprint  # Always swap same fingerprint

    f = cf.CuckooFilter(capacity=2, max_displacements=3)

    dummy = DummyBucket()
    # Replace all buckets by the dummy bucket so insert never succeeds
    f.buckets = [dummy, dummy]

    # Patch random.choice to always pick index 0
    monkeypatch.setattr(random, "choice", lambda x: x[0])

    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("full_bucket_test")


def test_repr_string_contains_parameters():
    f = cf.CuckooFilter(capacity=42, bucket_size=5, fingerprint_size=2)
    rep = repr(f)
    assert "capacity=42" in rep
    assert "fingerprint size=2" in rep
    assert "size=" in rep


def test_len_returns_size():
    f = cf.CuckooFilter(capacity=10)
    assert len(f) == 0
    f.insert("one")
    assert len(f) == 1
    f.insert("two")
    assert len(f) == 2
    f.delete("one")
    assert len(f) == 1


def test_contains_uses_alternate_indexing(monkeypatch):
    """
    Patch hashutils.hash_code and fingerprint so we can track calls and values.
    """

    calls = {"hash_code": [], "fingerprint": []}

    def fake_hash_code(value):
        calls["hash_code"].append(value)
        return 42

    def fake_fingerprint(item, size):
        calls["fingerprint"].append((item, size))
        # Returns a constant fingerprint byte string
        return b'\x01'

    monkeypatch.setattr(cf.hashutils, "hash_code", fake_hash_code)
    monkeypatch.setattr(cf.hashutils, "fingerprint", fake_fingerprint)

    f = cf.CuckooFilter(capacity=50)
    # Insert must call fingerprint and hash_code
    f.insert("myitem")
    assert calls["fingerprint"] != []
    assert calls["hash_code"] != []

    # Contains also calls them
    calls["fingerprint"].clear()
    calls["hash_code"].clear()
    f.contains("myitem")
    assert calls["fingerprint"] != []
    assert calls["hash_code"] != []


def test_delete_decrements_size_only_when_item_found():
    f = cf.CuckooFilter(capacity=5)
    f.insert("found")
    sz_before = len(f)
    assert f.delete("found") is True
    assert len(f) == sz_before - 1
    assert f.delete("notfound") is False
    assert len(f) == sz_before - 1


def test_insert_increases_size_only_once(monkeypatch):
    """
    Test that insert increases size by 1 even if fingerprint inserts in first or alternate bucket.
    """
    f = cf.CuckooFilter(capacity=5)

    count = {"inserts": 0}

    # Patch bucket.insert to track calls and simulate success on first call only
    orig_insert = bucket.Bucket.insert

    def insert_once(self, fingerprint):
        if count["inserts"] == 0:
            count["inserts"] += 1
            return True
        return False

    monkeypatch.setattr(bucket.Bucket, "insert", insert_once)

    # Insert an item, should increase size by one
    res = f.insert("val")
    assert res is True
    assert len(f) == 1


def test_contains_checks_both_buckets(monkeypatch):
    """
    Ensure contains() checks membership in both buckets.
    We patch buckets to control __contains__ behavior.
    """

    class FakeBucket:
        def __init__(self, fingerprints_in):
            self.fingerprints_in = fingerprints_in

        def __contains__(self, item):
            return item in self.fingerprints_in

    f = cf.CuckooFilter(capacity=3)

    # Prepare fingerprints
    fp = hashutils.fingerprint("item", f.fingerprint_size)

    i = f._get_index("item")
    j = f._get_alternate_index(i, fp)

    f.buckets[i] = FakeBucket(fingerprints_in=set())
    f.buckets[j] = FakeBucket(fingerprints_in={fp})

    # Contains should be True because fingerprint is in second bucket
    assert f.contains("item") is True


def test_insert_and_delete_consistency():
    f = cf.CuckooFilter(capacity=20)

    items = ["apple", "banana", "cherry", "date"]

    for item in items:
        assert f.insert(item) is True
    assert len(f) == len(items)

    for item in items:
        assert f.contains(item) is True

    for item in items:
        assert f.delete(item) is True
    assert len(f) == 0

    for item in items:
        assert f.contains(item) is False


def test_delete_twice_returns_false_second_time():
    f = cf.CuckooFilter(capacity=10)
    f.insert("value")
    assert f.delete("value") is True
    assert f.delete("value") is False


def test_insert_on_full_bucket_triggers_eviction(monkeypatch):
    """
    This test verifies that when both buckets are full, insert attempts evictions.
    We patch Bucket.insert and Bucket.swap.
    """

    class FullBucket:
        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            # Returns the fingerprint to simulate eviction swap
            return fingerprint

    f = cf.CuckooFilter(capacity=2, max_displacements=1)

    f.buckets = [FullBucket(), FullBucket()]

    monkeypatch.setattr(random, "choice", lambda x: x[0])

    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("someitem")


def test_fingerprint_size_effect_in_constructor():
    fp_size = 3
    f = cf.CuckooFilter(capacity=5, fingerprint_size=fp_size)
    assert f.fingerprint_size == fp_size


def test_get_index_and_get_alternate_index_consistent():
    f = cf.CuckooFilter(capacity=100)
    item = "abc123"
    fingerprint = hashutils.fingerprint(item, f.fingerprint_size)

    i = f._get_index(item)
    alt_i = f._get_alternate_index(i, fingerprint)

    # Alternate index should be different or equal but valid within capacity bounds
    assert 0 <= i < f.capacity
    assert 0 <= alt_i < f.capacity
    assert isinstance(i, int)
    assert isinstance(alt_i, int)


def test_insert_and_delete_edge_case_empty_filter():
    f = cf.CuckooFilter(capacity=1)
    val = "only"

    assert f.insert(val) is True
    assert f.contains(val) is True

    assert f.delete(val) is True
    assert f.contains(val) is False
    assert len(f) == 0


def test_delete_returns_false_when_bucket_delete_returns_false(monkeypatch):
    """
    Patch bucket.delete to always return False, check delete returns False.
    """
    f = cf.CuckooFilter(capacity=3)
    monkeypatch.setattr(bucket.Bucket, "delete", lambda self, fp: False)
    assert f.delete("any") is False


def test_insert_increases_size_and_returns_true_when_first_bucket_inserts(monkeypatch):
    called = {"insert_i": False, "insert_j": False}

    class TestBucket(bucket.Bucket):
        def insert(self, fingerprint):
            # First call returns True, others False
            if not called["insert_i"]:
                called["insert_i"] = True
                return True
            return False

        def swap(self, fingerprint):
            return fingerprint

    f = cf.CuckooFilter(capacity=3)
    # Replace buckets with TestBucket instances
    f.buckets = [TestBucket(size=4) for _ in range(f.capacity)]

    assert f.insert("a") is True
    assert len(f) == 1


def test_insert_increases_size_and_returns_true_when_second_bucket_inserts(monkeypatch):
    called = {"insert_i": False, "insert_j": False}

    class TestBucket(bucket.Bucket):
        def insert(self, fingerprint):
            # Fail first insert and succeed on second
            if not called["insert_i"]:
                called["insert_i"] = True
                return False
            elif not called["insert_j"]:
                called["insert_j"] = True
                return True
            return False

        def swap(self, fingerprint):
            return fingerprint

    f = cf.CuckooFilter(capacity=3)

    f.buckets = [TestBucket(size=4) for _ in range(f.capacity)]

    # Patch _get_index to always 0 and _get_alternate_index to always 1
    monkeypatch.setattr(f, "_get_index", lambda item: 0)
    monkeypatch.setattr(f, "_get_alternate_index", lambda index, fp: 1)

    assert f.insert("b") is True
    assert len(f) == 1

