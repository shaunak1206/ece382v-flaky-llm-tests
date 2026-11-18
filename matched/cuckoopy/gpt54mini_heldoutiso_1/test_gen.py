import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException
import cuckoopy.cuckoofilter as cf_module


class FakeBucket:
    def __init__(self, size):
        self.size = size
        self.items = []

    def insert(self, item):
        if len(self.items) < self.size:
            self.items.append(item)
            return True
        return False

    def delete(self, item):
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def swap(self, item):
        # Replace the first item and return the evicted item.
        evicted = self.items[0]
        self.items[0] = item
        return evicted

    def __contains__(self, item):
        return item in self.items


@pytest.fixture
def fake_hashes(monkeypatch):
    """
    Patch hashutils so tests can precisely control bucket selection.
    hash_code(item) -> item for ints, otherwise 0
    fingerprint(item, size) -> item itself if int, else 1-byte constant
    """
    monkeypatch.setattr(
        cf_module.hashutils,
        "hash_code",
        lambda item: item if isinstance(item, int) else 0,
    )
    monkeypatch.setattr(
        cf_module.hashutils,
        "fingerprint",
        lambda item, size: item if isinstance(item, int) else b"x" * size,
    )


@pytest.fixture
def fake_bucket_class(monkeypatch):
    monkeypatch.setattr(cf_module.bucket, "Bucket", FakeBucket)


def test_exception_class_inheritance_and_message():
    exc = CuckooFilterFullException("full")
    assert isinstance(exc, Exception)
    assert str(exc) == "full"


def test_init_creates_expected_number_of_buckets_and_sizes(fake_bucket_class):
    f = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2)
    assert len(f.buckets) == 5
    assert all(isinstance(b, FakeBucket) for b in f.buckets)
    assert all(b.size == 3 for b in f.buckets)
    assert f.capacity == 5
    assert f.bucket_size == 3
    assert f.fingerprint_size == 2
    assert f.size == 0


def test_repr_contains_key_fields(fake_bucket_class):
    f = CuckooFilter(capacity=7, bucket_size=2, fingerprint_size=4)
    r = repr(f)
    assert "<CuckooFilter:" in r
    assert "capacity=7" in r
    assert "size=0" in r
    assert "fingerprint size=4 byte(s)" in r


def test_len_tracks_insertions_and_deletions(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    assert len(f) == 0
    f.insert(1)
    assert len(f) == 1
    f.delete(1)
    assert len(f) == 0


def test_contains_uses_inserted_fingerprint(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    assert f.insert(3) is True
    assert 3 in f
    assert f.contains(3) is True


def test_contains_returns_false_for_missing_item(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    f.insert(3)
    assert f.contains(4) is False
    assert 4 not in f


def test_insert_places_item_in_primary_bucket_when_available(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    assert f.insert(1) is True
    assert 1 in f.buckets[1]
    assert f.size == 1


def test_insert_places_item_in_alternate_bucket_when_primary_full(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=1)
    # item 1 -> primary bucket 1, alternate bucket 0 (because hash_code(fingerprint)=1)
    assert f.insert(1) is True
    assert f.insert(11) is True  # also primary 1, alternate 0
    assert 1 in f.buckets[1]
    assert 11 in f.buckets[0]
    assert f.size == 2


def test_insert_uses_eviction_path_and_succeeds(fake_bucket_class, monkeypatch):
    # Configure to force eviction path:
    # capacity 2, bucket_size 1, both buckets filled with other fingerprints.
    monkeypatch.setattr(cf_module.hashutils, "hash_code", lambda item: item)
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", lambda item, size: item)
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])

    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=5)
    # Pre-fill buckets with distinct values.
    assert f.insert(0) is True
    assert f.insert(1) is True

    # Now inserting 2 should evict and eventually succeed.
    assert f.insert(2) is True
    assert f.size == 3
    assert f.contains(2) is True


def test_insert_raises_full_exception_when_max_displacements_exceeded(fake_bucket_class, monkeypatch):
    monkeypatch.setattr(cf_module.hashutils, "hash_code", lambda item: 0)
    monkeypatch.setattr(cf_module.hashutils, "fingerprint", lambda item, size: b"a")
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])

    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=0)
    assert f.insert("first") is True
    assert f.insert("second") is True

    with pytest.raises(CuckooFilterFullException) as excinfo:
        f.insert("third")

    assert "Filter is full" in str(excinfo.value)
    assert f.size == 2


def test_delete_returns_true_and_decrements_size(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    f.insert(5)
    assert f.delete(5) is True
    assert f.size == 0
    assert f.contains(5) is False


def test_delete_returns_false_for_missing_item(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    f.insert(5)
    assert f.delete(6) is False
    assert f.size == 1


def test_delete_removes_from_alternate_bucket_when_needed(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=1)
    f.insert(1)   # primary 1
    f.insert(11)  # primary 1, alternate 0
    assert 11 in f.buckets[0]
    assert f.delete(11) is True
    assert 11 not in f.buckets[0]
    assert f.size == 1


def test_multiple_insert_contains_delete_flow(fake_bucket_class, fake_hashes):
    f = CuckooFilter(capacity=10, bucket_size=2)
    items = [2, 4, 6]
    for item in items:
        assert f.insert(item) is True

    assert all(item in f for item in items)
    assert f.delete(4) is True
    assert 4 not in f
    assert 2 in f
    assert 6 in f
    assert len(f) == 2
