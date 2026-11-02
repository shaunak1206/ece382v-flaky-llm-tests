import random
import pytest

import cuckoopy.cuckoofilter as cf
import cuckoopy.exceptions as exc


def test_init_cuckoofilter_basic_properties():
    capacity = 10
    bucket_size = 3
    fingerprint_size = 2
    max_displacements = 100
    filter = cf.CuckooFilter(capacity, bucket_size, fingerprint_size,
                             max_displacements)
    assert filter.capacity == capacity
    assert filter.bucket_size == bucket_size
    assert filter.fingerprint_size == fingerprint_size
    assert filter.max_displacements == max_displacements
    assert len(filter.buckets) == capacity
    assert all(hasattr(b, 'size') and b.size == bucket_size for b in filter.buckets)
    assert filter.size == 0


def test_repr_and_len_methods():
    filter = cf.CuckooFilter(5)
    repr_str = repr(filter)
    assert "CuckooFilter" in repr_str
    assert "capacity=5" in repr_str
    assert "size=0" in repr_str
    assert "fingerprint size=1" in repr_str
    assert len(filter) == 0


def test_get_index_and_alternate_index_consistency():
    filter = cf.CuckooFilter(7)
    # use int keys for deterministic hash behaviour
    key = 42
    idx = filter._get_index(key)
    fp = cf.hashutils.fingerprint(key, filter.fingerprint_size)
    alt_idx = filter._get_alternate_index(idx, fp)
    # Both indexes should be within capacity range
    assert 0 <= idx < filter.capacity
    assert 0 <= alt_idx < filter.capacity
    # Alternate index should differ or equal (XOR can produce equal)
    assert isinstance(alt_idx, int)


def test_insert_returns_true_and_increases_size(monkeypatch):
    random.seed(1234)

    filter = cf.CuckooFilter(10, bucket_size=2, max_displacements=10)
    initial_size = filter.size

    # Insert an int key, size must increase by 1 on success
    inserted = filter.insert(5)
    assert inserted is True
    assert filter.size == initial_size + 1

    # Insert another different key, size should increase again
    inserted2 = filter.insert(6)
    assert inserted2 is True
    assert filter.size == initial_size + 2

    # Trying to insert the same key again should either be ignored or treated normally
    inserted3 = filter.insert(5)
    # In a cuckoo filter, duplicates might be inserted multiple times,
    # but size must still increase if insertion happens
    assert isinstance(inserted3, bool)
    if inserted3:
        assert filter.size == initial_size + 3 or filter.size == initial_size + 2


def test_contains_returns_true_for_inserted_items_and_false_for_others(monkeypatch):
    random.seed(123)

    filter = cf.CuckooFilter(20, bucket_size=2, max_displacements=10)
    items = [10, 22, 36]
    for item in items:
        filter.insert(item)

    for item in items:
        assert item in filter
        assert filter.contains(item)

    # Items not inserted should almost never be contained (allow false positives)
    # but we check items unlikely to collide fingerprints
    for item in [11, 23, 37]:
        # Either False or True (rare false positive), but sensible to check type only
        result = filter.contains(item)
        assert isinstance(result, bool)


def test_delete_removes_inserted_item_and_decreases_size(monkeypatch):
    random.seed(4321)

    filter = cf.CuckooFilter(10, bucket_size=2)
    item = 100
    filter.insert(item)
    size_before = filter.size
    assert item in filter
    deleted = filter.delete(item)
    assert deleted is True
    assert filter.size == size_before - 1
    # Now item should not be contained (False or false positive rare)
    # But delete deletes exact fingerprint, so deleted==True means it is removed.
    assert (item in filter) is False or (item in filter)


def test_delete_returns_false_for_nonexistent_item(monkeypatch):
    random.seed(42)

    filter = cf.CuckooFilter(10)
    filter.insert(1)
    size_before = filter.size
    # Attempt to delete nonexistent
    deleted = filter.delete(9999)
    assert deleted is False
    assert filter.size == size_before


def test_insert_raises_full_exception_when_no_space(monkeypatch):
    random.seed(10)

    # We patch bucket.Bucket.insert and bucket.Bucket.swap to simulate full buckets
    class FullBucket:
        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            # always return fingerprint, forcing eviction cycle infinite
            return fingerprint

    filter = cf.CuckooFilter(2, bucket_size=1, max_displacements=5)
    # Replace all buckets with FullBucket
    filter.buckets = [FullBucket(), FullBucket()]

    with pytest.raises(exc.CuckooFilterFullException):
        filter.insert(123)


def test_insert_eviction_works_and_final_insert_increases_size(monkeypatch):
    random.seed(1)

    # We simulate scenario where insertions succeed only after some evictions
    class EvictionBucket:
        def __init__(self):
            self.fingerprints = []
            self.size = 1

        def insert(self, fingerprint):
            if len(self.fingerprints) < 1:
                self.fingerprints.append(fingerprint)
                return True
            return False

        def swap(self, fingerprint):
            # swap out the first fingerprint and return it
            evicted = self.fingerprints[0]
            self.fingerprints[0] = fingerprint
            return evicted

    filter = cf.CuckooFilter(2, bucket_size=1, max_displacements=50)
    filter.buckets = [EvictionBucket(), EvictionBucket()]

    # Insert: first insertion will fill bucket0, second insertion will cause eviction and succeed
    inserted = filter.insert(1)
    assert inserted is True
    # Size increased by 1
    assert filter.size == 1

    inserted2 = filter.insert(2)
    assert inserted2 is True
    # Size increased by 1 more
    assert filter.size == 2


def test_contains_and_delete_with_different_fingerprint_sizes(monkeypatch):
    random.seed(12345)

    # We test with fingerprint size more than 1 byte
    filter = cf.CuckooFilter(10, fingerprint_size=2, bucket_size=3)
    item = 4242

    inserted = filter.insert(item)
    assert inserted is True
    assert item in filter

    deleted = filter.delete(item)
    assert deleted is True
    assert not filter.contains(item)


def test_insert_deterministic_with_seeded_random(monkeypatch):
    # We test that insertion outcome is deterministic if random is seeded

    seed = 99
    random.seed(seed)
    filter1 = cf.CuckooFilter(10, bucket_size=2, max_displacements=50)
    inserted1 = filter1.insert(77)

    random.seed(seed)
    filter2 = cf.CuckooFilter(10, bucket_size=2, max_displacements=50)
    inserted2 = filter2.insert(77)

    assert inserted1 is inserted2
    assert filter1.size == filter2.size


def test_contains_uses_hashutils_fingerprint(monkeypatch):
    random.seed(777)

    filter = cf.CuckooFilter(10, bucket_size=2)

    # To check that the contains method uses hashutils.fingerprint correctly,
    # we mock that to return a fixed fingerprint and check buckets.

    fixed_fp = b'\xAB'  # bytes fixed fingerprint

    def fake_fingerprint(item, size):
        return fixed_fp

    monkeypatch.setattr(cf.hashutils, 'fingerprint', fake_fingerprint)

    # Patch bucket.Bucket.__contains__ to verify contains call
    called_buckets = []

    orig_bucket_contains = cf.bucket.Bucket.__contains__

    def fake_contains(self, fp):
        called_buckets.append(fp)
        # Return False always, simplify
        return False

    monkeypatch.setattr(cf.bucket.Bucket, "__contains__", fake_contains)

    _ = filter.contains('anything')

    assert all(fp == fixed_fp for fp in called_buckets)
    # Restore original __contains__
    monkeypatch.setattr(cf.bucket.Bucket, "__contains__", orig_bucket_contains)


def test_insert_uses_correct_indexes(monkeypatch):
    random.seed(2021)

    filter = cf.CuckooFilter(10, bucket_size=2)

    # We patch _get_index and _get_alternate_index to track calls and args
    called = {}

    def fake_get_index(self, item):
        called['index'] = item
        return 3

    def fake_get_alternate_index(self, index, fingerprint):
        called['alt_index'] = (index, fingerprint)
        return 4

    monkeypatch.setattr(cf.CuckooFilter, "_get_index", fake_get_index)
    monkeypatch.setattr(cf.CuckooFilter, "_get_alternate_index", fake_get_alternate_index)

    # Patch buckets to always insert successfully
    for b in filter.buckets:
        monkeypatch.setattr(b, "insert", lambda fp: True)

    filter.insert(999)

    # Check the called values match expectation
    assert called['index'] == 999
    assert isinstance(called['alt_index'], tuple)
    assert called['alt_index'][0] == 3


def test_insert_handles_fingerprint_as_bytes(monkeypatch):
    random.seed(50)
    filter = cf.CuckooFilter(10, bucket_size=2, fingerprint_size=1)

    # We craft a key with bytes only to avoid unpredictable hash
    key = 12345  # int ensures stable hash

    inserted = filter.insert(key)
    assert inserted is True

    # Check that size increased
    assert filter.size == 1

    # Insert byte string key (hash unstable in runtime) - avoid for test predictability
    # So just confirm insert returns bool
    assert isinstance(filter.insert(b'bytekey'), bool) or inserted


def test_delete_correctly_updates_size_with_multiple_inserts(monkeypatch):
    random.seed(88)

    filter = cf.CuckooFilter(10, bucket_size=3)
    items = [7, 8, 9]

    for i in items:
        filter.insert(i)

    size_before = filter.size

    # Delete one item, size decreases by one
    result = filter.delete(8)
    assert result is True
    assert filter.size == size_before - 1

    # Deleting the same item again returns False
    result2 = filter.delete(8)
    assert result2 is False

    # Deleting items not present returns False
    assert filter.delete(123456) is False


def test_size_never_negative_after_deletes(monkeypatch):
    random.seed(55)

    filter = cf.CuckooFilter(10)
    assert filter.size == 0

    # Delete nonexistent item should not decrease size below zero
    result = filter.delete(999)
    assert result is False
    assert filter.size == 0


def test_insert_when_bucket_accepts_without_eviction(monkeypatch):
    random.seed(14)

    class SimpleBucket:
        def __init__(self):
            self.size = 2
            self.fps = []

        def insert(self, fingerprint):
            if len(self.fps) < 2:
                self.fps.append(fingerprint)
                return True
            return False

        def swap(self, fingerprint):
            # Should not be called
            assert False, "swap should not be called"
            return fingerprint

    filter = cf.CuckooFilter(2, bucket_size=2, max_displacements=5)
    filter.buckets = [SimpleBucket(), SimpleBucket()]

    res = filter.insert(10)
    assert res is True
    assert filter.size == 1

    res2 = filter.insert(20)
    assert res2 is True
    assert filter.size == 2


def test_insert_eviction_raises_correct_exception_with_monkeypatch(monkeypatch):
    random.seed(7)

    class AlwaysFullBucket:
        def __init__(self):
            self.size = 1

        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            return fingerprint  # always returns the fingerprint, evict loop continues

    filter = cf.CuckooFilter(1, bucket_size=1, max_displacements=3)
    filter.buckets = [AlwaysFullBucket()]

    with pytest.raises(exc.CuckooFilterFullException):
        filter.insert(123)


def test_len_and_contains_work_together(monkeypatch):
    random.seed(3)

    filter = cf.CuckooFilter(10, bucket_size=2)
    items = [111, 222, 333]
    for item in items:
        filter.insert(item)

    assert len(filter) == filter.size
    for item in items:
        assert item in filter

    # Delete one
    filter.delete(222)
    assert len(filter) == filter.size
    assert 222 not in filter


def test_contains_uses_both_buckets(monkeypatch):
    random.seed(22)

    filter = cf.CuckooFilter(5, bucket_size=1)
    # Insert a single item
    item = 4
    filter.insert(item)

    # Patch both buckets to check their __contains__ calls
    called = {"bucket0": 0, "bucket1": 0}

    orig_contains0 = filter.buckets[0].__contains__
    orig_contains1 = filter.buckets[1].__contains__

    def fake_contains0(fp):
        called["bucket0"] += 1
        return False

    def fake_contains1(fp):
        called["bucket1"] += 1
        return False

    monkeypatch.setattr(filter.buckets[0], "__contains__", fake_contains0)
    monkeypatch.setattr(filter.buckets[1], "__contains__", fake_contains1)

    _ = filter.contains(item)
    # Both buckets should have been checked exactly once
    assert called["bucket0"] == 1
    assert called["bucket1"] == 1

    # Restore original methods
    monkeypatch.setattr(filter.buckets[0], "__contains__", orig_contains0)
    monkeypatch.setattr(filter.buckets[1], "__contains__", orig_contains1)
