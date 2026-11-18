import pytest

from cuckoopy import cuckoofilter
from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def _patch_deterministic_hashing(monkeypatch):
    def hash_code(value):
        if isinstance(value, bytes):
            return sum(value)
        if isinstance(value, str):
            return sum(value.encode("utf-8"))
        return int(value)

    def fingerprint(item, size=1):
        if isinstance(item, bytes):
            base = item
        elif isinstance(item, str):
            base = item.encode("utf-8")
        else:
            base = str(item).encode("utf-8")
        seed = base[:1] or b"x"
        return seed * size

    monkeypatch.setattr(cuckoofilter.hashutils, "hash_code", hash_code)
    monkeypatch.setattr(cuckoofilter.hashutils, "fingerprint", fingerprint)


def test_exception_class_inheritance_and_message():
    exc = CuckooFilterFullException("Insert operation failed.")
    assert isinstance(exc, Exception)
    assert isinstance(exc, CuckooFilterFullException)
    assert str(exc) == "Insert operation failed."


def test_constructor_initializes_capacity_bucket_size_and_length():
    cf = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2, max_displacements=10)
    assert cf.capacity == 5
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf) == 0
    assert cf.size == 0
    assert len(cf.buckets) == 5


def test_repr_includes_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=7, bucket_size=4, fingerprint_size=2)
    assert repr(cf) == "<CuckooFilter: capacity=7, size=0, fingerprint size=2 byte(s)>"


def test_bucket_size_attribute_is_preserved():
    cf = CuckooFilter(capacity=4, bucket_size=8, fingerprint_size=1)
    assert cf.bucket_size == 8
    assert len(cf.buckets) == 4


def test_insert_returns_true_and_increments_length(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    assert cf.insert(1) is True
    assert len(cf) == 1
    assert cf.size == 1


def test_contains_returns_true_for_inserted_item(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    cf.insert(42)
    assert cf.contains(42) is True


def test_contains_returns_false_for_missing_item(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    cf.insert(1)
    assert cf.contains(2) is False
    assert (2 in cf) is False


def test_in_operator_delegates_to_contains(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    cf.insert("alpha")
    assert "alpha" in cf
    assert "beta" not in cf


def test_delete_existing_item_returns_true_and_decrements_length(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    cf.insert(5)
    assert cf.delete(5) is True
    assert len(cf) == 0
    assert cf.size == 0
    assert cf.contains(5) is False


def test_delete_missing_item_returns_false_and_does_not_change_length(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    cf.insert(1)
    assert cf.delete(2) is False
    assert len(cf) == 1
    assert cf.contains(1) is True


def test_multiple_inserts_and_deletes_keep_size_consistent(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    items = [10, 11, 12, 13]
    for item in items:
        assert cf.insert(item) is True

    assert len(cf) == 4
    assert all(item in cf for item in items)

    assert cf.delete(11) is True
    assert cf.delete(13) is True
    assert len(cf) == 2
    assert 11 not in cf
    assert 13 not in cf
    assert 10 in cf
    assert 12 in cf


def test_insert_can_handle_collisions_via_alternate_bucket(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)

    def constant_hash_code(value):
        if isinstance(value, bytes):
            return 1
        if isinstance(value, str):
            return 1
        return 1

    monkeypatch.setattr(cuckoofilter.hashutils, "hash_code", constant_hash_code)

    cf = CuckooFilter(capacity=4, bucket_size=4, fingerprint_size=1)
    assert cf.insert(1) is True
    assert cf.insert(2) is True
    assert len(cf) == 2
    assert cf.contains(1) is True
    assert cf.contains(2) is True


def test_insert_raises_full_exception_when_no_space_and_no_displacements(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)

    def constant_hash_code(value):
        if isinstance(value, bytes):
            return 0
        if isinstance(value, str):
            return 0
        return 0

    monkeypatch.setattr(cuckoofilter.hashutils, "hash_code", constant_hash_code)

    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    assert cf.insert(1) is True

    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert(2)

    assert "Filter is full" in str(excinfo.value)
    assert len(cf) == 1


def test_delete_after_reinsertion_keeps_membership_semantics(monkeypatch):
    _patch_deterministic_hashing(monkeypatch)
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=1)

    assert cf.insert("item") is True
    assert cf.delete("item") is True
    assert "item" not in cf

    assert cf.insert("item") is True
    assert "item" in cf
    assert len(cf) == 1
