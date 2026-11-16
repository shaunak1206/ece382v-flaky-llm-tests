"""Unit tests for cuckoopy.cuckoofilter.CuckooFilter and cuckoopy.exceptions."""

import pytest

from cuckoopy import exceptions
from cuckoopy.cuckoofilter import CuckooFilter


def fill_until_full(cf, items):
    """Insert items until the filter raises; return the number inserted."""
    inserted = 0
    for item in items:
        try:
            cf.insert(item)
        except exceptions.CuckooFilterFullException:
            break
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Construction and bucket_size
# ---------------------------------------------------------------------------

def test_init_stores_constructor_arguments():
    cf = CuckooFilter(capacity=8, bucket_size=3, fingerprint_size=2,
                      max_displacements=17)
    assert cf.capacity == 8
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 17
    assert len(cf.buckets) == 8
    assert len(cf) == 0


def test_default_bucket_size_is_four():
    cf = CuckooFilter(capacity=4)
    assert cf.bucket_size == 4


def test_bucket_size_limits_entries_per_bucket():
    # With capacity 1 every item maps to the single bucket, so the number of
    # successful inserts before the filter is full equals bucket_size.
    cf = CuckooFilter(capacity=1, bucket_size=2, max_displacements=0)
    assert cf.insert('a') is True
    assert cf.insert('b') is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('c')
    assert len(cf) == 2


def test_default_bucket_size_capacity_one_holds_four_items():
    cf = CuckooFilter(capacity=1, max_displacements=0)
    for item in ['w', 'x', 'y', 'z']:
        assert cf.insert(item) is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('overflow')
    assert len(cf) == 4


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_returns_true_and_increments_len():
    cf = CuckooFilter(capacity=16)
    assert cf.insert('apple') is True
    assert len(cf) == 1


def test_insert_multiple_distinct_items_increments_len_each_time():
    cf = CuckooFilter(capacity=64)
    for n, item in enumerate(['one', 'two', 'three', 'four', 'five'], start=1):
        cf.insert(item)
        assert len(cf) == n


def test_insert_same_item_twice_counts_twice():
    cf = CuckooFilter(capacity=16)
    cf.insert('dup')
    cf.insert('dup')
    assert len(cf) == 2


def test_insert_into_full_filter_raises_cuckoo_filter_full_exception():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert('first')
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('second')


def test_failed_insert_does_not_change_len():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert('first')
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('second')
    assert len(cf) == 1


def test_insert_eventually_raises_when_filter_saturates():
    cf = CuckooFilter(capacity=4, bucket_size=1, max_displacements=10)
    items = ['item-{}'.format(n) for n in range(1000)]
    inserted = fill_until_full(cf, items)
    assert inserted < len(items)
    assert len(cf) == inserted
    assert inserted <= cf.capacity * cf.bucket_size


# ---------------------------------------------------------------------------
# contains / __contains__
# ---------------------------------------------------------------------------

def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=16)
    assert cf.contains('nothing') is False
    assert ('nothing' in cf) is False


def test_contains_inserted_item_is_true():
    cf = CuckooFilter(capacity=16)
    cf.insert('present')
    assert cf.contains('present') is True
    assert 'present' in cf


def test_no_false_negatives_for_inserted_items():
    # Power-of-two capacity keeps the partial-key alternate-index mapping
    # symmetric, so every inserted item must still be found after evictions.
    cf = CuckooFilter(capacity=128, bucket_size=4, fingerprint_size=2)
    items = ['key-{}'.format(n) for n in range(300)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.contains(item) is True


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_inserted_item_returns_true_and_decrements_len():
    cf = CuckooFilter(capacity=16)
    cf.insert('gone')
    assert cf.delete('gone') is True
    assert len(cf) == 0
    assert cf.contains('gone') is False


def test_delete_missing_item_on_empty_filter_returns_false():
    cf = CuckooFilter(capacity=16)
    assert cf.delete('missing') is False
    assert len(cf) == 0


def test_delete_all_inserted_items_leaves_filter_empty():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    items = ['d-{}'.format(n) for n in range(20)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0


def test_delete_frees_space_for_new_insert():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert('old')
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert('new')
    assert cf.delete('old') is True
    assert cf.insert('new') is True
    assert cf.contains('new') is True


# ---------------------------------------------------------------------------
# CuckooFilterFullException
# ---------------------------------------------------------------------------

def test_cuckoo_filter_full_exception_is_exception_subclass():
    assert issubclass(exceptions.CuckooFilterFullException, Exception)


def test_cuckoo_filter_full_exception_preserves_message():
    err = exceptions.CuckooFilterFullException('Filter is full.')
    assert str(err) == 'Filter is full.'


def test_cuckoo_filter_full_exception_can_be_raised_and_caught():
    with pytest.raises(exceptions.CuckooFilterFullException) as excinfo:
        raise exceptions.CuckooFilterFullException('boom')
    assert 'boom' in str(excinfo.value)
