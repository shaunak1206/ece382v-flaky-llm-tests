import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture
def deterministic_hashes(monkeypatch):
    """Make hash/fingerprint behavior deterministic for tests."""
    from cuckoopy import hashutils

    mapping = {
        "alpha": (0, b"a"),
        "beta": (1, b"b"),
        "gamma": (0, b"c"),
        "delta": (0, b"d"),
        "epsilon": (0, b"e"),
        "zeta": (0, b"f"),
        "eta": (0, b"g"),
        "theta": (0, b"h"),
        "iota": (0, b"i"),
        "kappa": (0, b"j"),
        "lambda": (0, b"k"),
        "mu": (0, b"l"),
    }

    def fake_hash_code(item):
        if item in mapping:
            return mapping[item][0]
        if isinstance(item, (bytes, bytearray)):
            return sum(item)
        return 0

    def fake_fingerprint(item, size=1):
        if item in mapping:
            return mapping[item][1][:size]
        if isinstance(item, (bytes, bytearray)):
            return bytes(item[:size]).ljust(size, b"\x00")
        return b"x" * size

    monkeypatch.setattr(hashutils, "hash_code", fake_hash_code)
    monkeypatch.setattr(hashutils, "fingerprint", fake_fingerprint)
    return mapping


def test_init_sets_bucket_size_and_length_zero():
    cf = CuckooFilter(capacity=4, bucket_size=8, fingerprint_size=2, max_displacements=10)
    assert cf.capacity == 4
    assert cf.bucket_size == 8
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf) == 0
    assert cf.size == 0
    assert len(cf.buckets) == 4
    assert all(bucket.size == 8 for bucket in cf.buckets)


def test_repr_includes_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=3, bucket_size=2, fingerprint_size=4)
    assert repr(cf) == "<CuckooFilter: capacity=3, size=0, fingerprint size=4 byte(s)>"


def test_insert_returns_true_and_increments_size(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    assert cf.insert("alpha") is True
    assert len(cf) == 1
    assert cf.size == 1


def test_contains_true_after_insert_using_contains_method(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    cf.insert("alpha")
    assert cf.contains("alpha") is True


def test_contains_true_after_insert_using_in_operator(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    cf.insert("alpha")
    assert "alpha" in cf


def test_contains_false_for_missing_item(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    cf.insert("alpha")
    assert cf.contains("beta") is False
    assert "beta" not in cf


def test_delete_returns_true_for_inserted_item_and_decrements_size(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    cf.insert("alpha")
    assert cf.delete("alpha") is True
    assert len(cf) == 0
    assert cf.contains("alpha") is False


def test_delete_returns_false_for_missing_item(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    cf.insert("alpha")
    assert cf.delete("beta") is False
    assert len(cf) == 1


def test_insert_allows_multiple_items_and_len_tracks_count(deterministic_hashes):
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    assert cf.insert("alpha") is True
    assert cf.insert("beta") is True
    assert cf.insert("gamma") is True
    assert len(cf) == 3


def test_delete_only_removes_one_item_when_duplicates_inserted(deterministic_hashes):
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    assert cf.insert("alpha") is True
    assert cf.insert("alpha") is True
    assert len(cf) == 2
    assert cf.delete("alpha") is True
    assert len(cf) == 1
    assert cf.contains("alpha") is True


def test_contains_uses_bucket_size_configuration(deterministic_hashes):
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1)
    assert cf.bucket_size == 1
    assert cf.insert("alpha") is True
    assert cf.contains("alpha") is True


def test_full_exception_is_raised_when_insert_never_succeeds(monkeypatch, deterministic_hashes):
    from cuckoopy import cuckoofilter as cf_module

    class DummyBucket:
        def __init__(self, size):
            self.size = size

        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            return b"z"

        def __contains__(self, fingerprint):
            return False

        def delete(self, fingerprint):
            return False

    monkeypatch.setattr(cf_module.bucket, "Bucket", DummyBucket)
    monkeypatch.setattr(cf_module.random, "choice", lambda seq: seq[0])

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=1)
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("alpha")

    assert "Filter is full" in str(excinfo.value)


def test_full_exception_string_representation():
    exc = CuckooFilterFullException("Insert operation failed. Filter is full.")
    assert isinstance(exc, Exception)
    assert "Filter is full" in str(exc)
