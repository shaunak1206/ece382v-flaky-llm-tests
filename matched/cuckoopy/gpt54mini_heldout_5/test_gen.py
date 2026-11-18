import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_sets_capacity_bucket_size_and_zero_size():
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=3, max_displacements=7)

    assert cf.capacity == 8
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 3
    assert cf.max_displacements == 7
    assert cf.size == 0
    assert len(cf.buckets) == 8
    assert len(cf) == 0


def test_repr_includes_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=4, bucket_size=3, fingerprint_size=2)

    rep = repr(cf)

    assert "<CuckooFilter:" in rep
    assert "capacity=4" in rep
    assert "size=0" in rep
    assert "fingerprint size=2 byte(s)" in rep


def test_insert_success_updates_size_and_contains_item():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    assert cf.insert("hello") is True
    assert cf.size == 1
    assert len(cf) == 1
    assert cf.contains("hello") is True
    assert "hello" in cf


def test_contains_returns_false_for_missing_item():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    cf.insert("present")

    assert cf.contains("missing") is False
    assert "missing" not in cf


def test_delete_existing_item_returns_true_and_decrements_size():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    cf.insert("alpha")
    assert cf.size == 1

    assert cf.delete("alpha") is True
    assert cf.size == 0
    assert cf.contains("alpha") is False
    assert "alpha" not in cf


def test_delete_missing_item_returns_false_and_preserves_size():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    cf.insert("alpha")
    before = cf.size

    assert cf.delete("beta") is False
    assert cf.size == before
    assert cf.contains("alpha") is True


def test_insert_multiple_distinct_items_tracks_size():
    cf = CuckooFilter(capacity=32, bucket_size=4, fingerprint_size=1)

    items = ["a", "b", "c", "d", "e"]
    for item in items:
        assert cf.insert(item) is True

    assert cf.size == len(items)
    for item in items:
        assert cf.contains(item) is True


def test_bucket_size_argument_is_exposed_on_filter_and_used_for_buckets():
    cf = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=1)

    assert cf.bucket_size == 3
    assert len(cf.buckets) == 5
    assert all(bucket.size == 3 for bucket in cf.buckets)


def test_delete_after_two_inserts_only_removes_target_item():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    cf.insert("alpha")
    cf.insert("beta")

    assert cf.delete("alpha") is True
    assert cf.contains("alpha") is False
    assert cf.contains("beta") is True
    assert cf.size == 1


def test_contains_reflects_removed_item_but_preserves_other_inserted_items():
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    cf.insert("one")
    cf.insert("two")
    cf.insert("three")

    assert cf.delete("two") is True
    assert cf.contains("one") is True
    assert cf.contains("two") is False
    assert cf.contains("three") is True
    assert cf.size == 2


def test_insert_uses_alternate_bucket_when_primary_is_full(monkeypatch):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1)

    monkeypatch.setattr(cf.hashutils, "hash_code", lambda item: 0)
    monkeypatch.setattr(cf.hashutils, "fingerprint", lambda item, size: item)

    assert cf.insert("first") is True
    assert cf.insert("second") is True

    assert cf.size == 2
    assert cf.contains("first") is True
    assert cf.contains("second") is True


def test_insert_raises_full_exception_when_no_displacements_can_succeed(monkeypatch):
    class FakeBucket:
        def __init__(self, size):
            self.size = size

        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            return fingerprint

        def delete(self, fingerprint):
            return False

        def __contains__(self, fingerprint):
            return False

    monkeypatch.setattr("cuckoopy.cuckoofilter.bucket.Bucket", FakeBucket)
    monkeypatch.setattr(
        "cuckoopy.cuckoofilter.hashutils.hash_code",
        lambda item: 0
    )
    monkeypatch.setattr(
        "cuckoopy.cuckoofilter.hashutils.fingerprint",
        lambda item, size: "fp"
    )
    monkeypatch.setattr("cuckoopy.cuckoofilter.random.choice", lambda seq: seq[0])

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=0)

    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("anything")

    assert "Filter is full" in str(excinfo.value)


def test_cuckoo_filter_full_exception_is_a_normal_exception():
    exc = CuckooFilterFullException("Insert operation failed. Filter is full.")

    assert isinstance(exc, Exception)
    assert str(exc) == "Insert operation failed. Filter is full."
