import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_stores_parameters():
    cf = CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2,
                      max_displacements=7)
    assert cf.capacity == 10
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 7
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_init_defaults():
    cf = CuckooFilter(capacity=8)
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert len(cf) == 0


def test_empty_filter_does_not_contain_item():
    cf = CuckooFilter(capacity=16)
    assert cf.contains("hello") is False
    assert "hello" not in cf


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=16)
    assert cf.insert("apple") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("apple")
    assert cf.contains("apple") is True
    assert "apple" in cf


def test_delete_existing_item():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    cf.insert("apple")
    assert cf.delete("apple") is True
    assert cf.size == 0
    assert cf.contains("apple") is False


def test_delete_nonexistent_item_returns_false():
    cf = CuckooFilter(capacity=16)
    assert cf.delete("ghost") is False
    assert cf.size == 0


def test_delete_does_not_change_size_when_missing():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("present")
    before = cf.size
    # Empty-of-this-item deletion on a different, never inserted item;
    # even on a rare fingerprint collision the result is consistent with size.
    result = cf.delete("absent-item-xyz")
    assert cf.size == (before - 1 if result else before)


def test_duplicate_inserts_require_matching_deletes():
    cf = CuckooFilter(capacity=16, fingerprint_size=2)
    assert cf.insert("dup") is True
    assert cf.insert("dup") is True
    assert cf.size == 2

    assert cf.delete("dup") is True
    assert cf.size == 1
    assert cf.contains("dup") is True

    assert cf.delete("dup") is True
    assert cf.size == 0
    assert cf.contains("dup") is False
    assert cf.delete("dup") is False


def test_delete_one_item_keeps_other_items():
    cf = CuckooFilter(capacity=128, fingerprint_size=2)
    cf.insert("a")
    cf.insert("b")
    assert cf.delete("a") is True
    assert cf.contains("b") is True
    assert cf.size == 1


def test_no_false_negatives_for_many_items():
    cf = CuckooFilter(capacity=1000, bucket_size=4, fingerprint_size=2)
    items = ["item-%d" % n for n in range(200)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 200
    for item in items:
        assert item in cf


def test_insert_then_delete_all_items_empties_filter():
    cf = CuckooFilter(capacity=500, bucket_size=4, fingerprint_size=2)
    items = ["key-%d" % n for n in range(100)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert cf.size == 0
    for item in items:
        assert item not in cf


@pytest.mark.parametrize("bucket_size", [1, 2, 3, 5])
def test_bucket_size_limits_entries_in_single_bucket_filter(bucket_size):
    # With capacity 1 every item maps to the same (only) bucket,
    # so exactly bucket_size inserts can succeed.
    cf = CuckooFilter(capacity=1, bucket_size=bucket_size,
                      max_displacements=10)
    assert cf.bucket_size == bucket_size
    for n in range(bucket_size):
        assert cf.insert("item-%d" % n) is True
    assert cf.size == bucket_size
    with pytest.raises(CuckooFilterFullException):
        cf.insert("one-too-many")


def test_full_filter_raises_exception_with_message():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException, match="full"):
        cf.insert("second")


def test_full_exception_does_not_increase_size():
    cf = CuckooFilter(capacity=1, bucket_size=2, max_displacements=5)
    cf.insert("x")
    cf.insert("y")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("z")
    assert cf.size == 2


def test_zero_max_displacements_raises_when_buckets_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")


def test_full_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)
    with pytest.raises(Exception):
        raise CuckooFilterFullException("boom")


def test_repr_contains_state():
    cf = CuckooFilter(capacity=32, fingerprint_size=2)
    cf.insert("z")
    text = repr(cf)
    assert "CuckooFilter" in text
    assert "capacity=32" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text
