import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException
import cuckoopy.cuckoofilter as cf_module


class FakeBucket:
    def __init__(self, size, insert_result=False, swap_result=None):
        self.size = size
        self.insert_result = insert_result
        self.swap_result = swap_result if swap_result is not None else "evicted"
        self.items = []

    def insert(self, item):
        if self.insert_result and len(self.items) < self.size:
            self.items.append(item)
            return True
        return False

    def contains(self, item):
        return item in self.items

    def delete(self, item):
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def swap(self, item):
        old = self.swap_result
        if self.items:
            self.items[0] = item
        return old


def patch_hashing(monkeypatch, hash_map, fingerprint_map):
    monkeypatch.setattr(cf_module.hashutils, "hash_code", lambda x: hash_map[x])
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", lambda item, size: fingerprint_map[item])


def test_init_creates_buckets_with_requested_bucket_size(monkeypatch):
    created = []

    class RecordingBucket:
        def __init__(self, size):
            created.append(size)
            self.size = size

    monkeypatch.setattr(cf_module.bucket, "Bucket", RecordingBucket)

    f = CuckooFilter(capacity=5, bucket_size=7)
    assert f.capacity == 5
    assert f.bucket_size == 7
    assert len(f.buckets) == 5
    assert created == [7, 7, 7, 7, 7]


def test_len_starts_at_zero_and_repr_includes_key_fields():
    f = CuckooFilter(capacity=3, bucket_size=2, fingerprint_size=4)
    assert len(f) == 0
    rep = repr(f)
    assert "<CuckooFilter:" in rep
    assert "capacity=3" in rep
    assert "size=0" in rep
    assert "fingerprint size=4 byte(s)" in rep


def test_insert_success_into_primary_bucket(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    assert f.insert("a") is True
    assert len(f) == 1
    assert f.contains("a") is True


def test_insert_success_into_alternate_bucket_when_primary_full(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    assert f.buckets[0].insert("fp-a") is True
    assert f.insert("a") is True
    assert len(f) == 1
    assert f.contains("a") is True


def test_contains_true_when_fingerprint_is_in_primary_bucket(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    f.buckets[0].insert("fp-a")
    assert f.contains("a") is True


def test_contains_true_when_fingerprint_is_in_alternate_bucket(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    f.buckets[1].insert("fp-a")
    assert f.contains("a") is True


def test_contains_false_when_item_not_present(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1, "b": 0, "fp-b": 1}, {"a": "fp-a", "b": "fp-b"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    assert f.contains("a") is False
    assert ("a" in f) is False


def test_delete_true_decrements_size_from_primary_bucket(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    f.insert("a")
    assert len(f) == 1
    assert f.delete("a") is True
    assert len(f) == 0
    assert f.contains("a") is False


def test_delete_true_decrements_size_from_alternate_bucket(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    f.buckets[1].insert("fp-a")
    f.size = 1
    assert f.delete("a") is True
    assert len(f) == 0
    assert f.contains("a") is False


def test_delete_false_for_missing_item_does_not_change_size(monkeypatch):
    patch_hashing(monkeypatch, {"a": 0, "fp-a": 1}, {"a": "fp-a"})
    f = CuckooFilter(capacity=2, bucket_size=1)

    f.insert("a")
    assert len(f) == 1
    assert f.delete("missing") is False
    assert len(f) == 1
    assert f.contains("a") is True


def test_contains_uses_current_fingerprint_size(monkeypatch):
    calls = []

    def fake_fingerprint(item, size):
        calls.append((item, size))
        return "fp"

    monkeypatch.setattr(cf_module.hashutils, "hash_code", lambda x: 0)
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", fake_fingerprint)

    f = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=3)
    f.contains("x")
    assert calls == [("x", 3)]


def test_insert_raises_full_exception_when_no_space_and_max_displacements_exceeded(monkeypatch):
    class AlwaysFullBucket:
        def __init__(self, size):
            self.size = size

        def insert(self, item):
            return False

        def swap(self, item):
            return item

        def delete(self, item):
            return False

        def contains(self, item):
            return False

    monkeypatch.setattr(cf_module.bucket, "Bucket", AlwaysFullBucket)
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(cf_module.hashutils, "hash_code", lambda x: 0)
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", lambda item, size: "fp")

    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=1)

    with pytest.raises(CuckooFilterFullException) as excinfo:
        f.insert("x")

    assert "Filter is full" in str(excinfo.value)
    assert len(f) == 0


def test_insert_then_delete_round_trip_preserves_other_items(monkeypatch):
    patch_hashing(
        monkeypatch,
        {"a": 0, "fp-a": 1, "b": 1, "fp-b": 0},
        {"a": "fp-a", "b": "fp-b"},
    )
    f = CuckooFilter(capacity=2, bucket_size=2)

    assert f.insert("a") is True
    assert f.insert("b") is True
    assert len(f) == 2
    assert f.delete("a") is True
    assert len(f) == 1
    assert f.contains("a") is False
    assert f.contains("b") is True
