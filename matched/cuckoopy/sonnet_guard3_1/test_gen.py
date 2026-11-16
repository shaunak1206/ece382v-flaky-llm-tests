import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_defaults():
    random.seed(0)
    cf = CuckooFilter(capacity=10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_bucket_size_custom_value():
    random.seed(0)
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=2,
                      max_displacements=10)
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf.buckets) == 8


def test_repr_and_len_empty():
    random.seed(0)
    cf = CuckooFilter(capacity=10)
    assert len(cf) == 0
    assert repr(cf) == ('<CuckooFilter: capacity=10, size=0, '
                        'fingerprint size=1 byte(s)>')


def test_insert_returns_true_and_updates_size():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    assert cf.insert(1) is True
    assert cf.size == 1
    assert len(cf) == 1
    assert cf.insert(2) is True
    assert len(cf) == 2


def test_contains_after_insert_int():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    cf.insert(42)
    assert cf.contains(42) is True


def test_contains_after_insert_str():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    cf.insert("hello")
    assert cf.contains("hello") is True


def test_contains_on_empty_filter_is_false():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    for item in range(20):
        assert cf.contains(item) is False


def test_in_operator_matches_contains():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    cf.insert(7)
    assert 7 in cf
    assert 8 not in cf


def test_no_false_negatives_for_many_items():
    random.seed(0)
    cf = CuckooFilter(capacity=200, bucket_size=4, fingerprint_size=2)
    items = list(range(60))
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item) is True


def test_delete_existing_item():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    cf.insert(5)
    assert cf.delete(5) is True
    assert len(cf) == 0
    assert cf.contains(5) is False


def test_delete_missing_item_from_empty_filter():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    assert cf.delete(123) is False
    assert len(cf) == 0


def test_delete_twice_second_returns_false():
    random.seed(0)
    cf = CuckooFilter(capacity=100)
    cf.insert(9)
    assert cf.delete(9) is True
    assert cf.delete(9) is False
    assert cf.size == 0


def test_delete_all_inserted_items_returns_true_and_empties_filter():
    random.seed(0)
    cf = CuckooFilter(capacity=200, fingerprint_size=2)
    items = list(range(30))
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert cf.size == 0


def test_full_filter_raises_exception_single_slot():
    random.seed(0)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    assert cf.insert(1) is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert(2)
    assert cf.size == 1


def test_full_filter_raises_after_bucket_filled():
    random.seed(0)
    cf = CuckooFilter(capacity=1, bucket_size=4, max_displacements=10)
    for item in range(4):
        assert cf.insert(item) is True
    assert len(cf) == 4
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert(100)
    assert "full" in str(excinfo.value).lower()
    assert len(cf) == 4


def test_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)
    with pytest.raises(CuckooFilterFullException):
        raise CuckooFilterFullException("full")
