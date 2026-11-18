import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException
import cuckoopy.cuckoofilter as cf_module


class FakeBucket:
    def __init__(self, size=4, initial=None, insert_results=None, delete_results=None, swap_results=None):
        self.size = size
        self.items = list(initial or [])
        self._insert_results = list(insert_results or [])
        self._delete_results = list(delete_results or [])
        self._swap_results = list(swap_results or [])

    def insert(self, fingerprint):
        if self._insert_results:
            result = self._insert_results.pop(0)
            if result and fingerprint not in self.items and len(self.items) < self.size:
                self.items.append(fingerprint)
            return result
        if len(self.items) >= self.size:
            return False
        self.items.append(fingerprint)
        return True

    def delete(self, fingerprint):
        if self._delete_results:
            result = self._delete_results.pop(0)
            if result and fingerprint in self.items:
                self.items.remove(fingerprint)
            return result
        if fingerprint in self.items:
            self.items.remove(fingerprint)
            return True
        return False

    def swap(self, fingerprint):
        if self._swap_results:
            return self._swap_results.pop(0)
        if self.items:
            evicted = self.items[0]
            self.items[0] = fingerprint
            return evicted
        self.items.append(fingerprint)
        return fingerprint

    def __contains__(self, fingerprint):
        return fingerprint in self.items


@pytest.fixture
def deterministic_hashes(monkeypatch):
    def fake_hash_code(obj):
        if isinstance(obj, str) and obj.startswith("fp:"):
            # Keep alternate index stable but different from primary.
            return 1
        return {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "missing": 0,
        }.get(obj, 0)

    def fake_fingerprint(item, size):
        return f"fp:{item}:{size}"

    monkeypatch.setattr(cf_module.hashutils, "hash_code", fake_hash_code)
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", fake_fingerprint)


def test_cuckoofilterfull_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_cuckoofilterfull_exception_can_be_raised_and_caught():
    with pytest.raises(CuckooFilterFullException, match="filter full"):
        raise CuckooFilterFullException("filter full")


def test_init_sets_attributes_and_creates_buckets():
    cf = CuckooFilter(capacity=3, bucket_size=2, fingerprint_size=4, max_displacements=7)
    assert cf.capacity == 3
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 4
    assert cf.max_displacements == 7
    assert len(cf.buckets) == 3
    assert all(bucket.size == 2 for bucket in cf.buckets)
    assert len(cf) == 0


def test_repr_includes_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=2, bucket_size=3, fingerprint_size=5)
    rep = repr(cf)
    assert "capacity=2" in rep
    assert "size=0" in rep
    assert "fingerprint size=5 byte(s)" in rep


def test_insert_success_in_primary_bucket_increments_size(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2)]

    assert cf.insert("a") is True
    assert len(cf) == 1
    assert "fp:a:1" in cf.buckets[0]
    assert "a" in cf
    assert cf.contains("a") is True


def test_insert_success_in_alternate_bucket_when_primary_is_full(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1)
    cf.buckets = [
        FakeBucket(size=1, initial=["occupied"]),
        FakeBucket(size=1),
    ]

    assert cf.insert("a") is True
    assert len(cf) == 1
    assert "fp:a:1" in cf.buckets[1]
    assert cf.contains("a") is True


def test_contains_returns_false_for_missing_item(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2)]

    assert cf.contains("missing") is False
    assert ("missing" in cf) is False


def test_delete_returns_true_and_decrements_size(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    fp = "fp:a:1"
    cf.buckets = [FakeBucket(size=2, initial=[fp]), FakeBucket(size=2)]
    cf.size = 1

    assert cf.delete("a") is True
    assert len(cf) == 0
    assert cf.contains("a") is False


def test_delete_returns_false_for_missing_item_and_keeps_size(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2)]
    cf.size = 0

    assert cf.delete("missing") is False
    assert len(cf) == 0


def test_contains_checks_both_candidate_buckets(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    fp = "fp:a:1"
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2, initial=[fp])]

    assert cf.contains("a") is True


def test_insert_uses_eviction_path_when_both_buckets_are_full(deterministic_hashes, monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=1)
    cf.buckets = [
        FakeBucket(size=1, initial=["old0"], insert_results=[False], swap_results=["old0"]),
        FakeBucket(size=1, initial=["old1"], insert_results=[False, True]),
    ]
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])

    assert cf.insert("a") is True
    assert len(cf) == 1
    assert "fp:a:1" in cf.buckets[1] or "fp:a:1" in cf.buckets[0]


def test_insert_raises_full_exception_when_max_displacements_exhausted(
    deterministic_hashes, monkeypatch
):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=1)
    cf.buckets = [
        FakeBucket(size=1, initial=["old0"], insert_results=[False], swap_results=["old0"]),
        FakeBucket(size=1, initial=["old1"], insert_results=[False], swap_results=["old1"]),
    ]
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])

    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        cf.insert("a")
    assert len(cf) == 0


def test_len_tracks_number_of_successful_inserts_only(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2)]

    assert cf.insert("a") is True
    assert cf.insert("b") is True
    assert len(cf) == 2


def test_delete_after_insert_removes_membership(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)
    cf.buckets = [FakeBucket(size=2), FakeBucket(size=2)]
    cf.insert("a")

    assert cf.contains("a") is True
    assert cf.delete("a") is True
    assert cf.contains("a") is False
    assert len(cf) == 0


def test_bucket_size_attribute_is_preserved_and_used_by_created_buckets():
    cf = CuckooFilter(capacity=4, bucket_size=7, fingerprint_size=1)
    assert cf.bucket_size == 7
    assert len(cf.buckets) == 4
    assert all(bucket.size == 7 for bucket in cf.buckets)
