import pytest

import cuckoopy.cuckoofilter as cfmod
from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


class FakeBucket:
    def __init__(self, size):
        self.size = size
        self.items = []

    def insert(self, fingerprint):
        if len(self.items) >= self.size:
            return False
        self.items.append(fingerprint)
        return True

    def delete(self, fingerprint):
        if fingerprint in self.items:
            self.items.remove(fingerprint)
            return True
        return False

    def swap(self, fingerprint):
        # Replace first item and return the evicted one.
        evicted = self.items[0]
        self.items[0] = fingerprint
        return evicted

    def __contains__(self, fingerprint):
        return fingerprint in self.items


class NeverInsertBucket(FakeBucket):
    def insert(self, fingerprint):
        return False


def test_cuckoo_filter_full_exception_is_exception_and_string():
    exc = CuckooFilterFullException("boom")
    assert isinstance(exc, Exception)
    assert str(exc) == "boom"


def test_init_creates_correct_number_of_buckets_and_records_bucket_size(monkeypatch):
    created_sizes = []

    class RecordingBucket(FakeBucket):
        def __init__(self, size):
            created_sizes.append(size)
            super().__init__(size)

    monkeypatch.setattr(cfmod.bucket, "Bucket", RecordingBucket)

    f = CuckooFilter(capacity=5, bucket_size=3)
    assert len(f.buckets) == 5
    assert created_sizes == [3, 3, 3, 3, 3]
    assert f.bucket_size == 3
    assert f.size == 0
    assert len(f) == 0


def test_repr_includes_capacity_size_and_fingerprint_size(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    f = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=4)
    assert repr(f) == "<CuckooFilter: capacity=2, size=0, fingerprint size=4 byte(s)>"


def test_contains_uses_dunder_contains_and_contains_method(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=2, bucket_size=2)
    f.buckets[0].items.append("fp-x")

    assert f.contains("x") is True
    assert ("x" in f) is True
    assert f.contains("y") is False
    assert ("y" not in f) is True


def test_insert_places_item_in_primary_bucket_and_increments_size(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=3, bucket_size=2)
    assert f.insert("alpha") is True
    assert f.size == 1
    assert len(f) == 1
    assert "alpha" in f
    assert f.buckets[0].items == ["fp-alpha"]


def test_insert_uses_alternate_bucket_when_primary_is_full(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: "fp")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=2, bucket_size=1)
    f.buckets[0].items.append("existing")

    assert f.insert("alpha") is True
    assert f.size == 1
    assert f.buckets[1].items == ["fp"]


def test_insert_uses_secondary_bucket_when_primary_insert_fails(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 1)

    f = CuckooFilter(capacity=4, bucket_size=1)
    i = f._get_index("x")
    j = f._get_alternate_index(i, "fp-x")
    assert i != j
    f.buckets[i].items.append("occupied")

    assert f.insert("x") is True
    assert f.size == 1
    assert f.buckets[j].items == ["fp-x"]


def test_insert_with_eviction_succeeds_after_swap(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)
    monkeypatch.setattr(cfmod.random, "choice", lambda seq: seq[0])

    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=3)
    f.buckets[0].items.append("old0")
    f.buckets[1].items.append("old1")

    assert f.insert("new") is True
    assert f.size == 1
    assert f.buckets[0].items == ["fp-new"]
    assert f.buckets[1].items == ["old0"]


def test_insert_raises_full_exception_when_all_displacements_fail(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", NeverInsertBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: "fp")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)
    monkeypatch.setattr(cfmod.random, "choice", lambda seq: seq[0])

    f = CuckooFilter(capacity=2, bucket_size=0, max_displacements=2)

    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        f.insert("alpha")

    assert f.size == 0


def test_delete_removes_from_primary_bucket_and_decrements_size(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=2, bucket_size=2)
    f.buckets[0].items.append("fp-alpha")
    f.size = 1

    assert f.delete("alpha") is True
    assert f.size == 0
    assert "alpha" not in f
    assert f.buckets[0].items == []


def test_delete_removes_from_alternate_bucket_when_primary_missing(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: "fp")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=2, bucket_size=2)
    f.buckets[1].items.append("fp")
    f.size = 1

    assert f.delete("alpha") is True
    assert f.size == 0
    assert f.buckets[1].items == []


def test_delete_returns_false_and_does_not_change_size_when_item_missing(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "fingerprint", lambda item, size: f"fp-{item}")
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 0)

    f = CuckooFilter(capacity=2, bucket_size=2)
    f.size = 2

    assert f.delete("missing") is False
    assert f.size == 2


def test_get_index_and_alternate_index_are_modulo_capacity(monkeypatch):
    monkeypatch.setattr(cfmod.bucket, "Bucket", FakeBucket)
    monkeypatch.setattr(cfmod.hashutils, "hash_code", lambda item: 17 if item == "item" else 5)

    f = CuckooFilter(capacity=4, bucket_size=1)
    idx = f._get_index("item")
    alt = f._get_alternate_index(idx, "fingerprint")

    assert idx == 1
    assert alt == (1 ^ 5) % 4
