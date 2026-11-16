"""
Unit tests for cuckoopy.cuckoofilter.CuckooFilter and
cuckoopy.exceptions.CuckooFilterFullException.

Notes on determinism:
- Every test seeds the `random` module at the start of its body, because
  CuckooFilter.insert calls random.choice on the eviction path.
- Items are ints, whose builtin hash() is fixed across runs. No str or bytes
  keys are used, so PYTHONHASHSEED has no effect.
- Each test builds its own filter. No module or global state is changed
  except through the monkeypatch fixture, which restores it afterwards.
- Assertions on membership of non-inserted items are avoided, because with
  fingerprint_size=1 a fingerprint collision can produce false positives.
"""

import random

import pytest

from cuckoopy import cuckoofilter
from cuckoopy import exceptions


def test_bucket_size_is_stored_on_filter():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=3)
    assert cf.bucket_size == 3


def test_default_bucket_size_is_four():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.bucket_size == 4


def test_one_bucket_object_per_capacity_slot():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=7, bucket_size=2)
    assert len(cf.buckets) == 7


def test_new_filter_has_zero_len():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert len(cf) == 0


def test_contains_on_empty_filter_is_false():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.contains(42) is False


def test_delete_on_empty_filter_returns_false():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.delete(42) is False
    assert len(cf) == 0


def test_insert_returns_true_on_success():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.insert(1) is True


def test_insert_then_contains_is_true():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=10)
    cf.insert(1)
    assert cf.contains(1) is True


def test_insert_increments_len_by_one_each_time():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    for n, item in enumerate([1, 2, 3], start=1):
        cf.insert(item)
        assert len(cf) == n


def test_in_operator_matches_contains():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    cf.insert(5)
    assert (5 in cf) is True
    assert (5 in cf) == cf.contains(5)


def test_many_inserted_items_are_all_contained():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    items = list(range(10))
    for item in items:
        assert cf.insert(item) is True
    for item in items:
        assert cf.contains(item) is True
    assert len(cf) == len(items)


def test_delete_inserted_item_returns_true_and_decrements_len():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    cf.insert(7)
    assert cf.delete(7) is True
    assert len(cf) == 0


def test_contains_is_false_after_deleting_only_item():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    cf.insert(7)
    cf.delete(7)
    assert cf.contains(7) is False


def test_second_delete_of_same_only_item_returns_false():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=100)
    cf.insert(7)
    assert cf.delete(7) is True
    assert cf.delete(7) is False
    assert len(cf) == 0


def test_full_exception_raised_when_no_displacements_allowed():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1,
                                   max_displacements=0)
    assert cf.insert(1) is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(2)


def test_full_exception_leaves_len_unchanged():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1,
                                   max_displacements=5)
    cf.insert(1)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(2)
    assert len(cf) == 1


def test_full_exception_with_monkeypatched_random_choice(monkeypatch):
    random.seed(0)
    # Pin the eviction choice so the displacement loop is fully deterministic.
    # monkeypatch restores random.choice after this test, even on failure.
    monkeypatch.setattr(random, 'choice', lambda seq: seq[0])
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1,
                                   max_displacements=3)
    cf.insert(1)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(2)
    assert cf.contains(1) is True


def test_bucket_size_two_allows_two_items_in_single_bucket():
    random.seed(0)
    # With capacity 1, both candidate buckets are the same bucket 0.
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=2,
                                   max_displacements=0)
    assert cf.insert(1) is True
    assert cf.insert(2) is True
    assert len(cf) == 2


def test_bucket_size_two_rejects_third_item_in_single_bucket():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=2,
                                   max_displacements=0)
    cf.insert(1)
    cf.insert(2)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(3)
    assert len(cf) == 2


def test_contains_still_true_for_stored_item_after_full_error():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1,
                                   max_displacements=0)
    cf.insert(1)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(2)
    assert cf.contains(1) is True


def test_full_exception_is_an_exception_subclass():
    random.seed(0)
    assert issubclass(exceptions.CuckooFilterFullException, Exception)


def test_full_exception_message_mentions_filter_full():
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=1, bucket_size=1,
                                   max_displacements=0)
    cf.insert(1)
    with pytest.raises(exceptions.CuckooFilterFullException,
                       match='Filter is full'):
        cf.insert(2)
