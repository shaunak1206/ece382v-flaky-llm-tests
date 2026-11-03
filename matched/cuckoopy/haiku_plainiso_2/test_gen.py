"""Unit tests for CuckooFilter insert/contains/delete, bucket_size, and the full exception."""

import pytest

from cuckoopy import hashutils
from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def _patch_hash_code(monkeypatch, index_for_string):
    """Pin bucket indices: strings map through index_for_string, and integer
    fingerprints pass through unchanged so the XOR alternate-index math is exact."""

    def fake_hash_code(data):
        if isinstance(data, str):
            return index_for_string(data)
        return data

    monkeypatch.setattr(hashutils, "hash_code", fake_hash_code)


def _stored_fingerprints(cuckoo):
    return [fp for bucket in cuckoo.buckets for fp in bucket.bucket]


def _name_with_distinct_alternate(primary, capacity, fingerprint_size):
    for n in range(200):
        name = "candidate-%d" % n
        fp = hashutils.fingerprint(name, fingerprint_size)
        alternate = (primary ^ fp) % capacity
        if alternate != primary:
            return name, alternate
    raise AssertionError("no candidate with a distinct alternate bucket")


def test_default_bucket_size_is_four():
    cuckoo = CuckooFilter(capacity=8)
    assert cuckoo.bucket_size == 4
    assert all(bucket.size == 4 for bucket in cuckoo.buckets)


def test_bucket_size_is_applied_to_every_bucket():
    cuckoo = CuckooFilter(capacity=10, bucket_size=2)
    assert cuckoo.bucket_size == 2
    assert len(cuckoo.buckets) == 10
    assert all(bucket.size == 2 for bucket in cuckoo.buckets)


def test_new_filter_is_empty():
    cuckoo = CuckooFilter(capacity=8)
    assert len(cuckoo) == 0
    assert "anything" not in cuckoo
    assert all(len(bucket) == 0 for bucket in cuckoo.buckets)


def test_repr_reports_capacity_size_and_fingerprint_width():
    cuckoo = CuckooFilter(capacity=8, fingerprint_size=2)
    assert repr(cuckoo) == "<CuckooFilter: capacity=8, size=0, fingerprint size=2 byte(s)>"


def test_insert_returns_true_and_increments_len():
    cuckoo = CuckooFilter(capacity=64, fingerprint_size=4)
    assert cuckoo.insert("one") is True
    assert len(cuckoo) == 1
    assert cuckoo.insert("two") is True
    assert len(cuckoo) == 2


def test_inserted_items_are_contained():
    cuckoo = CuckooFilter(capacity=64, fingerprint_size=4)
    items = ["apple", "banana", "cherry", "date", "elderberry"]
    for item in items:
        cuckoo.insert(item)
    for item in items:
        assert cuckoo.contains(item) is True


def test_in_operator_matches_contains():
    cuckoo = CuckooFilter(capacity=64, fingerprint_size=4)
    cuckoo.insert("present")
    assert ("present" in cuckoo) is True
    assert ("present" in cuckoo) == cuckoo.contains("present")
    assert ("absent-item" in cuckoo) is False


def test_contains_does_not_modify_filter():
    cuckoo = CuckooFilter(capacity=16, fingerprint_size=4)
    cuckoo.insert("x")
    before = [list(bucket.bucket) for bucket in cuckoo.buckets]
    for _ in range(3):
        assert cuckoo.contains("x") is True
        cuckoo.contains("y")
    after = [list(bucket.bucket) for bucket in cuckoo.buckets]
    assert after == before
    assert len(cuckoo) == 1


def test_delete_removes_inserted_item():
    cuckoo = CuckooFilter(capacity=64, fingerprint_size=4)
    cuckoo.insert("doomed")
    assert cuckoo.delete("doomed") is True
    assert cuckoo.contains("doomed") is False
    assert len(cuckoo) == 0


def test_delete_missing_item_returns_false():
    cuckoo = CuckooFilter(capacity=8)
    assert cuckoo.delete("ghost") is False
    assert len(cuckoo) == 0


def test_delete_same_item_twice_succeeds_once():
    cuckoo = CuckooFilter(capacity=8, fingerprint_size=4)
    cuckoo.insert("once")
    assert cuckoo.delete("once") is True
    assert cuckoo.delete("once") is False
    assert len(cuckoo) == 0


def test_stored_fingerprint_matches_hashutils_width():
    cuckoo = CuckooFilter(capacity=32, fingerprint_size=2)
    cuckoo.insert("gamma")
    expected = hashutils.fingerprint("gamma", 2)
    assert 0 <= expected < 2 ** 16
    assert expected in _stored_fingerprints(cuckoo)


def test_fingerprint_size_one_stores_single_byte_values():
    cuckoo = CuckooFilter(capacity=64, bucket_size=4, fingerprint_size=1)
    names = ["name-%d" % n for n in range(10)]
    for name in names:
        assert cuckoo.insert(name) is True
    stored = _stored_fingerprints(cuckoo)
    assert len(stored) == 10
    assert all(0 <= fp <= 255 for fp in stored)


def test_insert_prefers_primary_bucket(monkeypatch):
    _patch_hash_code(monkeypatch, lambda name: 2)
    cuckoo = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=4)
    assert cuckoo.insert("alpha") is True
    assert hashutils.fingerprint("alpha", 4) in cuckoo.buckets[2].bucket
    assert all(len(bucket) == 0
               for index, bucket in enumerate(cuckoo.buckets) if index != 2)


def test_insert_falls_back_to_alternate_bucket(monkeypatch):
    _patch_hash_code(monkeypatch, lambda name: 2)
    cuckoo = CuckooFilter(capacity=8, bucket_size=1, fingerprint_size=4)
    assert cuckoo.insert("occupant") is True

    newcomer, alternate = _name_with_distinct_alternate(2, 8, 4)
    assert cuckoo.insert(newcomer) is True
    assert hashutils.fingerprint(newcomer, 4) in cuckoo.buckets[alternate].bucket
    assert cuckoo.contains(newcomer) is True


def test_delete_finds_item_in_alternate_bucket(monkeypatch):
    _patch_hash_code(monkeypatch, lambda name: 2)
    cuckoo = CuckooFilter(capacity=8, bucket_size=1, fingerprint_size=4)
    assert cuckoo.insert("occupant") is True

    newcomer, alternate = _name_with_distinct_alternate(2, 8, 4)
    assert cuckoo.insert(newcomer) is True
    assert cuckoo.delete(newcomer) is True
    assert len(cuckoo.buckets[alternate]) == 0
    assert cuckoo.contains("occupant") is True
    assert len(cuckoo) == 1


def test_bucket_size_caps_entries_per_bucket():
    cuckoo = CuckooFilter(capacity=1, bucket_size=3, fingerprint_size=4,
                          max_displacements=10)
    for name in ("one", "two", "three"):
        assert cuckoo.insert(name) is True
    assert len(cuckoo.buckets[0]) == 3
    with pytest.raises(CuckooFilterFullException):
        cuckoo.insert("four")
    assert len(cuckoo) == 3


def test_insert_into_full_filter_raises_full_exception():
    cuckoo = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                          max_displacements=5)
    assert cuckoo.insert("first") is True
    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        cuckoo.insert("second")


def test_failed_insert_leaves_len_unchanged():
    cuckoo = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                          max_displacements=5)
    cuckoo.insert("first")
    with pytest.raises(CuckooFilterFullException):
        cuckoo.insert("second")
    assert len(cuckoo) == 1


def test_zero_max_displacements_gives_up_without_swapping():
    cuckoo = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                          max_displacements=0)
    assert cuckoo.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cuckoo.insert("second")
    assert cuckoo.contains("first") is True
    assert len(cuckoo) == 1


def test_full_exception_is_an_exception():
    assert issubclass(CuckooFilterFullException, Exception)


def test_many_inserts_remain_findable():
    cuckoo = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=4)
    names = ["entry-%d" % n for n in range(40)]
    for name in names:
        assert cuckoo.insert(name) is True
    assert len(cuckoo) == 40
    for name in names:
        assert cuckoo.contains(name) is True


def test_bulk_delete_keeps_other_items():
    cuckoo = CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=4)
    names = ["entry-%d" % n for n in range(30)]
    for name in names:
        assert cuckoo.insert(name) is True
    for name in names[::2]:
        assert cuckoo.delete(name) is True
    assert len(cuckoo) == 15
    for name in names[1::2]:
        assert cuckoo.contains(name) is True
    for name in names[::2]:
        assert cuckoo.contains(name) is False


def test_insert_rejects_non_string_items():
    cuckoo = CuckooFilter(capacity=8)
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        cuckoo.insert(42)
    assert len(cuckoo) == 0
