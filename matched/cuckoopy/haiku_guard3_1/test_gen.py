"""Unit tests for cuckoopy.cuckoofilter.CuckooFilter and the
CuckooFilterFullException class in cuckoopy.exceptions."""

import random

import pytest

from cuckoopy import cuckoofilter
from cuckoopy import exceptions
from cuckoopy import hashutils


def test_default_bucket_size_is_four():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    assert f.bucket_size == 4


def test_custom_bucket_size_is_stored():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8, bucket_size=7)
    assert f.bucket_size == 7


def test_one_bucket_is_created_per_capacity_slot():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=10, bucket_size=3)
    assert len(f.buckets) == 10


def test_new_filter_is_empty_and_contains_nothing():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    assert len(f) == 0
    assert f.contains(1) is False
    assert (1 in f) is False


def test_insert_returns_true_and_increments_len():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    assert f.insert(1) is True
    assert len(f) == 1


def test_inserted_item_is_contained():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(42)
    assert f.contains(42) is True


def test_contains_dunder_matches_contains_method():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(7)
    assert (7 in f) == f.contains(7)
    assert (7 in f) is True


def test_len_dunder_matches_size_attribute():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(1)
    f.insert(2)
    assert len(f) == f.size == 2


def test_delete_inserted_item_returns_true_and_decrements_len():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(5)
    assert f.delete(5) is True
    assert len(f) == 0


def test_item_is_not_contained_after_its_only_copy_is_deleted():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(5)
    f.delete(5)
    assert f.contains(5) is False


def test_delete_from_empty_filter_returns_false():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    assert f.delete(99) is False
    assert len(f) == 0


def test_second_delete_of_single_item_returns_false():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=8)
    f.insert(3)
    assert f.delete(3) is True
    assert f.delete(3) is False
    assert len(f) == 0


def test_several_distinct_items_insert_contain_and_delete():
    # With at most 3 items and bucket_size=4, no bucket pair can ever be
    # full, so every insert succeeds on the first attempt and every item
    # keeps its own stored copy until it is deleted.
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=100, bucket_size=4)
    items = [11, 22, 33]
    for item in items:
        assert f.insert(item) is True
    assert len(f) == 3
    for item in items:
        assert f.contains(item) is True
    for item in items:
        assert f.delete(item) is True
    assert len(f) == 0


def test_repr_reports_capacity_size_and_fingerprint_size():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=10, fingerprint_size=1)
    f.insert(1)
    f.insert(2)
    text = repr(f)
    assert "capacity=10" in text
    assert "size=2" in text
    assert "fingerprint size=1 byte(s)" in text


def test_fingerprint_size_is_stored_and_filter_still_works():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=16, fingerprint_size=2)
    assert f.fingerprint_size == 2
    assert f.insert(100) is True
    assert f.contains(100) is True
    assert f.delete(100) is True


def test_single_slot_filter_raises_full_exception_on_second_insert():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1)
    assert f.insert(1) is True
    with pytest.raises(exceptions.CuckooFilterFullException, match="full"):
        f.insert(2)


def test_failed_insert_does_not_change_len():
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1)
    f.insert(1)
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert(2)
    assert len(f) == 1


def test_successful_inserts_never_exceed_slot_count():
    # Only capacity * bucket_size = 6 slots exist, so at most 6 inserts can
    # succeed out of 50 attempts; the rest must raise.
    random.seed(0)
    f = cuckoofilter.CuckooFilter(capacity=3, bucket_size=2)
    successes = 0
    failures = 0
    for item in range(50):
        try:
            assert f.insert(item) is True
            successes += 1
        except exceptions.CuckooFilterFullException:
            failures += 1
    assert successes <= 6
    assert failures >= 1
    assert len(f) == successes


def test_bucket_size_limits_items_when_all_items_collide(monkeypatch):
    # Force every item to hash to bucket 0 (index and alternate index are
    # both 0), so only bucket_size items fit and eviction cannot help.
    random.seed(0)
    monkeypatch.setattr(hashutils, "hash_code", lambda item: 0)
    f = cuckoofilter.CuckooFilter(capacity=2, bucket_size=3)
    for item in range(3):
        assert f.insert(item) is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert(3)
    assert len(f) == 3


def test_zero_max_displacements_fails_immediately_on_collision(monkeypatch):
    random.seed(0)
    monkeypatch.setattr(hashutils, "hash_code", lambda item: 0)
    f = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1,
                                  max_displacements=0)
    assert f.insert(1) is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert(2)
    assert len(f) == 1


def test_full_exception_is_an_exception_subclass():
    assert issubclass(exceptions.CuckooFilterFullException, Exception)


def test_full_exception_carries_message_and_can_be_caught_as_exception():
    with pytest.raises(Exception) as excinfo:
        raise exceptions.CuckooFilterFullException("Filter is full.")
    assert isinstance(excinfo.value, exceptions.CuckooFilterFullException)
    assert str(excinfo.value) == "Filter is full."
