import random

import pytest

import cuckoopy.cuckoofilter as cuckoo_mod
from cuckoopy.exceptions import CuckooFilterFullException


def _patch_deterministic_hashes(monkeypatch):
    # Use fixed, integer-based hashes to avoid dependence on Python's salted hash()
    # and to make bucket placement deterministic.
    monkeypatch.setattr(cuckoo_mod.hashutils, "hash_code", lambda item: int(item))
    monkeypatch.setattr(
        cuckoo_mod.hashutils,
        "fingerprint",
        lambda item, size: int(item) + 1,
    )


def test_init_sets_attributes_and_empty_length():
    cf = cuckoo_mod.CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2, max_displacements=7)

    assert cf.capacity == 5
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 7
    assert len(cf) == 0
    assert cf.size == 0
    assert len(cf.buckets) == 5


def test_insert_single_item_succeeds_and_contains_returns_true(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    assert cf.insert(10) is True

    assert len(cf) == 1
    assert 10 in cf
    assert cf.contains(10) is True


def test_contains_returns_false_for_absent_item(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    cf.insert(10)

    assert cf.contains(11) is False
    assert 11 not in cf


def test_delete_existing_item_returns_true_and_removes_item(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    cf.insert(10)

    assert cf.delete(10) is True
    assert len(cf) == 0
    assert cf.contains(10) is False
    assert 10 not in cf


def test_delete_absent_item_returns_false_and_does_not_change_size(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    cf.insert(10)

    assert cf.delete(11) is False
    assert len(cf) == 1
    assert cf.contains(10) is True


def test_multiple_inserts_increment_size_and_all_items_are_contained(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=16, bucket_size=2, fingerprint_size=1)
    items = [1, 2, 3, 4]

    for item in items:
        assert cf.insert(item) is True

    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item) is True


def test_bucket_size_attribute_is_preserved(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=4, bucket_size=7, fingerprint_size=1)
    assert cf.bucket_size == 7
    assert all(bucket.size == 7 for bucket in cf.buckets)


def test_repr_includes_capacity_size_and_fingerprint_size(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=3, bucket_size=2, fingerprint_size=4)
    cf.insert(1)

    r = repr(cf)
    assert "<CuckooFilter:" in r
    assert "capacity=3" in r
    assert "size=1" in r
    assert "fingerprint size=4 byte(s)" in r


def test_insert_into_full_filter_raises_cuckoo_filter_full_exception(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    assert cf.insert(1) is True

    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        cf.insert(2)


def test_failed_insert_does_not_change_size(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    cf.insert(1)

    with pytest.raises(CuckooFilterFullException):
        cf.insert(2)

    assert len(cf) == 1
    assert cf.contains(1) is True


def test_delete_same_item_twice_second_delete_returns_false(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    cf.insert(10)

    assert cf.delete(10) is True
    assert cf.delete(10) is False
    assert len(cf) == 0


def test_contains_reflects_delete_result(monkeypatch):
    random.seed(0)
    _patch_deterministic_hashes(monkeypatch)

    cf = cuckoo_mod.CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    cf.insert(42)

    assert cf.contains(42) is True
    assert cf.delete(42) is True
    assert cf.contains(42) is False
