import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_defaults():
    random.seed(1234)
    cf = CuckooFilter(capacity=10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_init_custom_parameters():
    random.seed(1234)
    cf = CuckooFilter(capacity=7, bucket_size=2, fingerprint_size=2,
                      max_displacements=11)
    assert cf.capacity == 7
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 11
    assert len(cf.buckets) == 7


def test_bucket_size_is_exposed_per_filter():
    random.seed(1234)
    small = CuckooFilter(capacity=5, bucket_size=1)
    large = CuckooFilter(capacity=5, bucket_size=8)
    assert small.bucket_size == 1
    assert large.bucket_size == 8


def test_len_and_repr_of_empty_filter():
    random.seed(1234)
    cf = CuckooFilter(capacity=10, fingerprint_size=2)
    assert len(cf) == 0
    assert repr(cf) == ('<CuckooFilter: capacity=10, size=0, '
                        'fingerprint size=2 byte(s)>')


def test_insert_returns_true_and_increments_size():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    assert cf.insert(42) is True
    assert cf.size == 1
    assert len(cf) == 1
    assert 'size=1' in repr(cf)


def test_contains_on_empty_filter_is_false():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    assert cf.contains(1) is False
    assert cf.contains('abc') is False
    assert 5 not in cf


def test_no_false_negatives_for_int_items():
    random.seed(1234)
    cf = CuckooFilter(capacity=1000)
    items = list(range(20))
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == len(items)
    for item in items:
        assert cf.contains(item) is True
        assert item in cf


def test_no_false_negatives_for_str_and_bytes_items():
    random.seed(1234)
    cf = CuckooFilter(capacity=1000, fingerprint_size=2)
    items = ['apple', 'banana', 'cherry', 'date']
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.contains(item) is True
    cf.insert('raw-bytes')
    assert 'raw-bytes' in cf


def test_delete_existing_item():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    cf.insert(7)
    assert cf.delete(7) is True
    assert cf.size == 0
    assert cf.contains(7) is False


def test_delete_from_empty_filter_returns_false():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    assert cf.delete(7) is False
    assert cf.size == 0


def test_delete_twice_second_returns_false():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    cf.insert(99)
    assert cf.delete(99) is True
    assert cf.delete(99) is False
    assert cf.size == 0


def test_duplicate_inserts_need_matching_deletes():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    assert cf.insert(5) is True
    assert cf.insert(5) is True
    assert cf.size == 2
    assert cf.delete(5) is True
    assert cf.size == 1
    assert cf.contains(5) is True
    assert cf.delete(5) is True
    assert cf.size == 0
    assert cf.contains(5) is False


def test_insert_after_delete_works():
    random.seed(1234)
    cf = CuckooFilter(capacity=100)
    cf.insert(3)
    cf.delete(3)
    assert cf.insert(3) is True
    assert cf.contains(3) is True
    assert cf.size == 1


def test_single_bucket_filter_holds_exactly_bucket_size_items():
    random.seed(1234)
    cf = CuckooFilter(capacity=1, bucket_size=3, max_displacements=10)
    for item in (1, 2, 3):
        assert cf.insert(item) is True
    assert cf.size == 3
    for item in (1, 2, 3):
        assert cf.contains(item) is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert(4)


def test_full_filter_does_not_count_failed_insert():
    random.seed(1234)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    assert cf.insert(1) is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert(2)
    assert cf.size == 1
    assert len(cf) == 1


def test_zero_max_displacements_raises_when_buckets_full():
    random.seed(1234)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert(1)
    with pytest.raises(CuckooFilterFullException):
        cf.insert(2)


def test_same_item_inserted_too_often_raises_full():
    random.seed(1234)
    bucket_size = 2
    cf = CuckooFilter(capacity=50, bucket_size=bucket_size,
                      max_displacements=10)
    # An item can occupy at most its two candidate buckets, so
    # 2 * bucket_size + 1 identical inserts must overflow.
    with pytest.raises(CuckooFilterFullException):
        for _ in range(2 * bucket_size + 1):
            cf.insert(123)


def test_full_exception_message_and_type():
    random.seed(1234)
    assert issubclass(CuckooFilterFullException, Exception)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=3)
    cf.insert(1)
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert(2)
    assert 'full' in str(excinfo.value).lower()


def test_exception_can_be_raised_and_caught_as_exception():
    random.seed(1234)
    with pytest.raises(Exception):
        raise CuckooFilterFullException('boom')
    assert str(CuckooFilterFullException('boom')) == 'boom'


def test_insert_and_delete_many_items_returns_to_empty():
    random.seed(1234)
    cf = CuckooFilter(capacity=1000)
    items = list(range(30))
    for item in items:
        cf.insert(item)
    assert cf.size == 30
    for item in items:
        assert cf.delete(item) is True
    assert cf.size == 0
    assert len(cf) == 0
