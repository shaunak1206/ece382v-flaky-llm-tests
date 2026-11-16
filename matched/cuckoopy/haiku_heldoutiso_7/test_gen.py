"""
Unit tests for cuckoopy.cuckoofilter.CuckooFilter and
cuckoopy.exceptions.CuckooFilterFullException.
"""

import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture(autouse=True)
def _seed_random():
    # insert() uses random.choice for eviction; seed for reproducibility.
    random.seed(12345)
    yield


# ---------------------------------------------------------------------------
# Construction and bucket_size
# ---------------------------------------------------------------------------

def test_init_stores_parameters():
    cf = CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2,
                      max_displacements=42)
    assert cf.capacity == 10
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 42


def test_init_default_parameters():
    cf = CuckooFilter(capacity=8)
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf) == 0


def test_init_creates_one_bucket_per_capacity_slot():
    cf = CuckooFilter(capacity=7, bucket_size=2)
    assert len(cf.buckets) == 7


def test_bucket_size_custom_value_is_kept():
    cf = CuckooFilter(capacity=5, bucket_size=6)
    assert cf.bucket_size == 6


def test_bucket_size_limits_entries_in_single_bucket_filter():
    # With capacity=1 every item maps to the same single bucket, so the
    # filter holds exactly bucket_size entries before it reports full.
    cf = CuckooFilter(capacity=1, bucket_size=3)
    assert cf.insert('a') is True
    assert cf.insert('b') is True
    assert cf.insert('c') is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert('d')
    assert len(cf) == 3


# ---------------------------------------------------------------------------
# insert / contains / __contains__ / __len__
# ---------------------------------------------------------------------------

def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=16)
    assert cf.insert('apple') is True
    assert len(cf) == 1
    assert cf.size == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('apple')
    assert cf.contains('apple') is True


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=16)
    assert cf.contains('apple') is False


def test_dunder_contains_matches_contains():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('pear')
    assert ('pear' in cf) is cf.contains('pear') is True
    assert ('plum' in cf) is cf.contains('plum')


def test_inserting_same_item_twice_counts_twice():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('dup')
    cf.insert('dup')
    assert len(cf) == 2
    assert cf.contains('dup') is True


def test_many_inserts_are_all_found():
    cf = CuckooFilter(capacity=256, bucket_size=4, fingerprint_size=4)
    items = ['item-{}'.format(i) for i in range(50)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 50
    for item in items:
        assert cf.contains(item) is True


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_inserted_item_returns_true_and_decrements_size():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('kiwi')
    assert cf.delete('kiwi') is True
    assert len(cf) == 0


def test_delete_removes_membership():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('kiwi')
    cf.delete('kiwi')
    assert cf.contains('kiwi') is False


def test_delete_absent_item_returns_false_and_keeps_size():
    cf = CuckooFilter(capacity=16)
    assert cf.delete('ghost') is False
    assert len(cf) == 0


def test_delete_one_of_two_duplicates_keeps_membership():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert('dup')
    cf.insert('dup')
    assert cf.delete('dup') is True
    assert len(cf) == 1
    assert cf.contains('dup') is True


def test_delete_does_not_affect_other_items():
    cf = CuckooFilter(capacity=256, fingerprint_size=4)
    cf.insert('keep-me')
    cf.insert('remove-me')
    assert cf.delete('remove-me') is True
    assert cf.contains('keep-me') is True
    assert len(cf) == 1


# ---------------------------------------------------------------------------
# CuckooFilterFullException
# ---------------------------------------------------------------------------

def test_insert_raises_full_exception_when_no_space():
    # Single slot total: second distinct insert must fail.
    cf = CuckooFilter(capacity=1, bucket_size=1)
    assert cf.insert('first') is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert('second')


def test_full_exception_does_not_increment_size():
    cf = CuckooFilter(capacity=1, bucket_size=1)
    cf.insert('first')
    with pytest.raises(CuckooFilterFullException):
        cf.insert('second')
    assert len(cf) == 1


def test_full_exception_is_an_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_full_exception_message_mentions_full():
    cf = CuckooFilter(capacity=1, bucket_size=1)
    cf.insert('first')
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert('second')
    assert 'full' in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_reflects_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=12, fingerprint_size=2)
    cf.insert('x')
    text = repr(cf)
    assert 'capacity=12' in text
    assert 'size=1' in text
    assert 'fingerprint size=2' in text
