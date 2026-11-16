import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_defaults():
    cf = CuckooFilter(capacity=10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_init_custom_parameters():
    cf = CuckooFilter(capacity=20, bucket_size=2, fingerprint_size=2,
                      max_displacements=10)
    assert cf.capacity == 20
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf.buckets) == 20


def test_repr_and_len_on_empty_filter():
    cf = CuckooFilter(capacity=10)
    assert len(cf) == 0
    assert repr(cf) == ('<CuckooFilter: capacity=10, size=0, '
                        'fingerprint size=1 byte(s)>')


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100)
    assert cf.insert('hello') is True
    assert cf.size == 1
    assert len(cf) == 1
    assert 'size=1' in repr(cf)


def test_contains_after_insert():
    cf = CuckooFilter(capacity=100)
    cf.insert('hello')
    assert cf.contains('hello') is True


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=100)
    assert cf.contains('anything') is False
    assert 'anything' not in cf


def test_in_operator_matches_contains():
    cf = CuckooFilter(capacity=100)
    cf.insert('abc')
    assert 'abc' in cf
    assert ('abc' in cf) == cf.contains('abc')


def test_no_false_negatives_for_many_items():
    cf = CuckooFilter(capacity=200, bucket_size=4, fingerprint_size=2)
    items = ['item-%d' % n for n in range(50)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 50
    for item in items:
        assert cf.contains(item) is True


def test_delete_existing_item():
    cf = CuckooFilter(capacity=100)
    cf.insert('hello')
    assert cf.delete('hello') is True
    assert cf.size == 0
    assert cf.contains('hello') is False


def test_delete_missing_item_returns_false_and_keeps_size():
    cf = CuckooFilter(capacity=100)
    assert cf.delete('missing') is False
    assert cf.size == 0


def test_delete_only_removes_target_item():
    cf = CuckooFilter(capacity=200, fingerprint_size=2)
    cf.insert('keep')
    cf.insert('remove')
    assert cf.delete('remove') is True
    assert cf.size == 1
    assert cf.contains('keep') is True


def test_reinsert_after_delete():
    cf = CuckooFilter(capacity=100)
    cf.insert('x')
    cf.delete('x')
    assert cf.insert('x') is True
    assert cf.contains('x') is True
    assert cf.size == 1


def test_insert_raises_when_filter_full_single_slot():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    assert cf.insert('first') is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert('second')
    assert cf.size == 1


def test_bucket_size_limits_number_of_entries():
    # With a single bucket, every item maps to it, so exactly bucket_size
    # items fit before the filter reports it is full.
    bucket_size = 3
    cf = CuckooFilter(capacity=1, bucket_size=bucket_size,
                      max_displacements=5)
    for n in range(bucket_size):
        assert cf.insert('item-%d' % n) is True
    assert cf.size == bucket_size
    with pytest.raises(CuckooFilterFullException):
        cf.insert('overflow')
    assert cf.size == bucket_size


def test_larger_bucket_size_holds_more_items():
    cf = CuckooFilter(capacity=1, bucket_size=6, max_displacements=5)
    for n in range(6):
        cf.insert('value-%d' % n)
    assert len(cf) == 6


def test_full_exception_message_and_type():
    assert issubclass(CuckooFilterFullException, Exception)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=3)
    cf.insert('a')
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert('b')
    assert 'full' in str(excinfo.value).lower()


def test_exception_can_be_raised_and_caught_directly():
    with pytest.raises(CuckooFilterFullException, match='boom'):
        raise CuckooFilterFullException('boom')
