"""
Unit tests for cuckoopy.cuckoofilter.CuckooFilter and
cuckoopy.exceptions.CuckooFilterFullException.
"""

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


# ---------------------------------------------------------------------------
# Construction and bucket_size
# ---------------------------------------------------------------------------

def test_default_bucket_size_is_four():
    cf = CuckooFilter(capacity=10)
    assert cf.bucket_size == 4


def test_custom_bucket_size_is_stored():
    cf = CuckooFilter(capacity=10, bucket_size=2)
    assert cf.bucket_size == 2


def test_number_of_buckets_matches_capacity():
    cf = CuckooFilter(capacity=7, bucket_size=3)
    assert len(cf.buckets) == 7


def test_new_filter_is_empty():
    cf = CuckooFilter(capacity=10)
    assert len(cf) == 0
    assert cf.size == 0


def test_repr_reports_state():
    cf = CuckooFilter(capacity=10, fingerprint_size=1)
    assert repr(cf) == '<CuckooFilter: capacity=10, size=0, ' \
                       'fingerprint size=1 byte(s)>'


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    assert cf.insert('apple') is True
    assert len(cf) == 1


def test_insert_multiple_items_increments_size_each_time():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    items = ['apple', 'banana', 'cherry', 'date']
    for count, item in enumerate(items, start=1):
        assert cf.insert(item) is True
        assert len(cf) == count


def test_inserted_item_is_contained():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    cf.insert('apple')
    assert cf.contains('apple')
    assert 'apple' in cf


def test_insert_same_item_twice_stores_two_copies():
    cf = CuckooFilter(capacity=1, bucket_size=4, fingerprint_size=4)
    assert cf.insert('dup') is True
    assert cf.insert('dup') is True
    assert len(cf) == 2
    assert cf.contains('dup')


def test_insert_into_full_filter_raises_and_keeps_size():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=5)
    assert cf.insert('first') is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert('second')
    assert len(cf) == 1


def test_filling_filter_eventually_raises_full_exception():
    cf = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=4,
                      max_displacements=10)
    with pytest.raises(CuckooFilterFullException):
        for n in range(100):
            cf.insert('item-%d' % n)
    # Total capacity is capacity * bucket_size = 4 slots.
    assert len(cf) <= 4


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------

def test_contains_false_on_empty_filter():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    assert cf.contains('ghost') is False
    assert ('ghost' in cf) is False


def test_contains_false_for_absent_item_in_populated_filter():
    cf = CuckooFilter(capacity=1000, fingerprint_size=4)
    for item in ['a', 'b', 'c']:
        cf.insert(item)
    # With a 4-byte fingerprint and only 3 items, a false positive here
    # would be astronomically unlikely.
    assert cf.contains('definitely-not-inserted-xyz') is False


def test_contains_is_stable_across_repeated_calls():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    cf.insert('stable')
    assert all(cf.contains('stable') for _ in range(5))


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_inserted_item_returns_true_and_decrements_size():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    cf.insert('apple')
    assert cf.delete('apple') is True
    assert len(cf) == 0
    assert not cf.contains('apple')


def test_delete_absent_item_returns_false_and_keeps_size():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    cf.insert('apple')
    assert cf.delete('pear') is False
    assert len(cf) == 1
    assert cf.contains('apple')


def test_delete_from_empty_filter_returns_false():
    cf = CuckooFilter(capacity=100, fingerprint_size=4)
    assert cf.delete('nothing') is False
    assert len(cf) == 0


def test_delete_one_of_duplicate_copies_keeps_item_present():
    cf = CuckooFilter(capacity=1, bucket_size=4, fingerprint_size=4)
    cf.insert('dup')
    cf.insert('dup')
    assert cf.delete('dup') is True
    assert len(cf) == 1
    assert cf.contains('dup')
    assert cf.delete('dup') is True
    assert len(cf) == 0
    assert not cf.contains('dup')


def test_insert_after_delete_succeeds_in_full_filter():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=5)
    cf.insert('first')
    with pytest.raises(CuckooFilterFullException):
        cf.insert('second')
    assert cf.delete('first') is True
    assert cf.insert('second') is True
    assert cf.contains('second')


# ---------------------------------------------------------------------------
# CuckooFilterFullException
# ---------------------------------------------------------------------------

def test_full_exception_is_an_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_full_exception_message_from_insert():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                      max_displacements=3)
    cf.insert('first')
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert('second')
    assert 'Filter is full' in str(excinfo.value)


def test_full_exception_can_be_raised_and_caught_directly():
    with pytest.raises(CuckooFilterFullException, match='custom'):
        raise CuckooFilterFullException('custom message')
