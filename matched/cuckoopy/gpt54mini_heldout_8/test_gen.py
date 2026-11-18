import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def _patch_hashing(monkeypatch, index_map, fp_map):
    from cuckoopy import hashutils

    def fake_hash_code(obj):
        if isinstance(obj, (bytes, bytearray)):
            return obj[0]
        return index_map[obj]

    def fake_fingerprint(item, size):
        return bytes([fp_map[item]]) * size

    monkeypatch.setattr(hashutils, "hash_code", fake_hash_code)
    monkeypatch.setattr(hashutils, "fingerprint", fake_fingerprint)


def test_exception_class_is_subclass_of_exception():
    assert issubclass(CuckooFilterFullException, Exception)


def test_init_sets_bucket_size_and_other_attributes():
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=3, max_displacements=17)
    assert cf.capacity == 8
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 3
    assert cf.max_displacements == 17
    assert len(cf.buckets) == 8
    assert len(cf) == 0


def test_repr_includes_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=5)
    assert repr(cf) == "<CuckooFilter: capacity=4, size=0, fingerprint size=5 byte(s)>"


def test_insert_successful_on_primary_bucket(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0}, {"a": 1})
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)

    assert cf.insert("a") is True
    assert len(cf) == 1
    assert cf.contains("a") is True


def test_insert_successful_on_alternate_bucket_when_primary_is_full(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0, "b": 0}, {"a": 1, "b": 1})
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1)

    assert cf.insert("a") is True
    assert cf.insert("b") is True
    assert len(cf) == 2
    assert cf.contains("a") is True
    assert cf.contains("b") is True


def test_contains_returns_false_for_missing_item(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0, "x": 0}, {"a": 1, "x": 0})
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)

    cf.insert("a")
    assert cf.contains("x") is False
    assert ("x" in cf) is False


def test_contains_uses_dunder_contains(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0}, {"a": 1})
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)

    cf.insert("a")
    assert ("a" in cf) is True


def test_delete_successful_removes_item_and_decrements_size(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0}, {"a": 1})
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)

    cf.insert("a")
    assert cf.delete("a") is True
    assert len(cf) == 0
    assert cf.contains("a") is False


def test_delete_returns_false_for_missing_item_and_does_not_change_size(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0, "x": 1}, {"a": 1, "x": 0})
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)

    cf.insert("a")
    assert cf.delete("x") is False
    assert len(cf) == 1
    assert cf.contains("a") is True


def test_bucket_size_parameter_controls_bucket_capacity(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0, "b": 0}, {"a": 1, "b": 1})
    cf = CuckooFilter(capacity=2, bucket_size=2, fingerprint_size=1)

    assert cf.bucket_size == 2
    assert cf.insert("a") is True
    assert cf.insert("b") is True
    assert len(cf) == 2
    assert cf.contains("a") is True
    assert cf.contains("b") is True


def test_insert_raises_full_exception_when_filter_is_full(monkeypatch):
    _patch_hashing(
        monkeypatch,
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 1},
    )
    monkeypatch.setattr("cuckoopy.cuckoofilter.random.choice", lambda seq: seq[0])

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=0)
    assert cf.insert("a") is True
    assert cf.insert("b") is True

    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("c")
    assert "Filter is full" in str(excinfo.value)
    assert len(cf) == 2


def test_insert_raises_full_exception_after_max_displacements(monkeypatch):
    _patch_hashing(
        monkeypatch,
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 1},
    )
    monkeypatch.setattr("cuckoopy.cuckoofilter.random.choice", lambda seq: seq[1])

    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=2)
    assert cf.insert("a") is True
    assert cf.insert("b") is True

    with pytest.raises(CuckooFilterFullException):
        cf.insert("c")


def test_delete_after_insertion_via_alternate_bucket(monkeypatch):
    _patch_hashing(monkeypatch, {"a": 0, "b": 0}, {"a": 1, "b": 1})
    cf = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1)

    cf.insert("a")
    cf.insert("b")

    assert cf.delete("b") is True
    assert cf.contains("b") is False
    assert len(cf) == 1
    assert cf.contains("a") is True
