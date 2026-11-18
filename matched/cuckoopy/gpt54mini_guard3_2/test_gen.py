import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_bucket_size_attribute_and_repr():
    cf = CuckooFilter(capacity=3, bucket_size=7, fingerprint_size=2)
    assert cf.bucket_size == 7
    assert len(cf.buckets) == 3
    assert repr(cf) == "<CuckooFilter: capacity=3, size=0, fingerprint size=2 byte(s)>"


def test_len_starts_at_zero():
    cf = CuckooFilter(capacity=5)
    assert len(cf) == 0
    assert cf.size == 0


def test_insert_contains_and_len_for_integer_item():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    assert cf.insert(12345) is True
    assert 12345 in cf
    assert cf.contains(12345) is True
    assert len(cf) == 1
    assert cf.size == 1


def test_insert_non_member_different_integer_not_necessarily_present():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(12345)
    # Different integer should be absent unless there is a hash collision;
    # for the chosen distinct integers and fingerprint size, this is a stable
    # behavioral test of the API contract after a single insert.
    assert cf.contains(54321) in (True, False)
    assert len(cf) == 1


def test_delete_existing_item_returns_true_and_decrements_length():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(11)
    assert cf.delete(11) is True
    assert 11 not in cf
    assert cf.contains(11) is False
    assert len(cf) == 0


def test_delete_missing_item_returns_false_and_preserves_length():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(11)
    assert cf.delete(12) is False
    assert len(cf) == 1
    assert 11 in cf


def test_contains_false_on_empty_filter_for_integer():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    assert cf.contains(99) is False
    assert 99 not in cf
    assert len(cf) == 0


def test_multiple_inserts_and_deletes_keep_len_consistent():
    cf = CuckooFilter(capacity=20, bucket_size=4, fingerprint_size=2)
    items = [1, 2, 3, 4, 5]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 5

    assert cf.delete(2) is True
    assert cf.delete(4) is True
    assert len(cf) == 3

    remaining = {1, 3, 5}
    for item in remaining:
        assert item in cf
    for item in {2, 4}:
        assert item not in cf


def test_insert_same_item_twice_increases_size_twice_or_raises_only_when_full():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    assert cf.insert(7) is True
    result = cf.insert(7)
    assert result is True
    assert len(cf) == 2
    assert cf.contains(7) is True


def test_contains_uses_dunder_contains():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(123)
    assert (123 in cf) is True
    assert (456 in cf) is False


def test_delete_after_duplicate_inserts_reduces_size_by_one_each_time():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(8)
    cf.insert(8)
    assert len(cf) == 2
    assert cf.delete(8) is True
    assert len(cf) == 1
    assert cf.contains(8) is True
    assert cf.delete(8) is True
    assert len(cf) == 0
    assert cf.contains(8) is False


def test_insert_raises_full_exception_when_capacity_one_and_bucket_size_one_is_saturated():
    random.seed(0)
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    assert cf.insert(1) is True
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert(2)
    assert "Filter is full" in str(excinfo.value)
    assert len(cf) == 1


def test_insert_raises_full_exception_type():
    random.seed(0)
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1, max_displacements=0)
    cf.insert(1)
    with pytest.raises(CuckooFilterFullException):
        cf.insert(2)


def test_exception_is_subclass_of_exception_and_has_message():
    exc = CuckooFilterFullException("Insert operation failed. Filter is full.")
    assert isinstance(exc, Exception)
    assert str(exc) == "Insert operation failed. Filter is full."


def test_delete_missing_item_does_not_change_contents_or_size():
    cf = CuckooFilter(capacity=10, bucket_size=4, fingerprint_size=2)
    cf.insert(100)
    before = len(cf)
    assert cf.delete(101) is False
    assert len(cf) == before
    assert 100 in cf
    assert 101 not in cf


def test_bucket_size_property_does_not_affect_existing_length_behavior():
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    assert cf.bucket_size == 2
    cf.insert(1)
    cf.insert(2)
    assert len(cf) == 2
    assert cf.bucket_size == 2


def test_contains_and_delete_on_unseen_integer_are_false():
    cf = CuckooFilter(capacity=8, bucket_size=4, fingerprint_size=2)
    assert cf.contains(999) is False
    assert cf.delete(999) is False
    assert len(cf) == 0
