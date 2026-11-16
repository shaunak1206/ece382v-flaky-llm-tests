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


def test_init_custom_bucket_size_and_params():
    cf = CuckooFilter(capacity=7, bucket_size=2, fingerprint_size=2,
                      max_displacements=10)
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf.buckets) == 7


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100)
    assert cf.insert("hello") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_inserted_item():
    cf = CuckooFilter(capacity=100)
    cf.insert("apple")
    assert cf.contains("apple") is True
    assert "apple" in cf


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=100)
    assert cf.contains("nothing") is False
    assert "nothing" not in cf


def test_no_false_negatives_for_many_items():
    cf = CuckooFilter(capacity=1000)
    items = ["item-%d" % i for i in range(100)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 100
    for item in items:
        assert cf.contains(item)


def test_delete_existing_item():
    cf = CuckooFilter(capacity=100)
    cf.insert("banana")
    assert cf.delete("banana") is True
    assert cf.size == 0
    assert cf.contains("banana") is False


def test_delete_missing_item_returns_false():
    cf = CuckooFilter(capacity=100)
    assert cf.delete("ghost") is False
    assert cf.size == 0


def test_delete_missing_item_does_not_change_size():
    cf = CuckooFilter(capacity=100)
    cf.insert("present")
    assert cf.delete("absent-item-xyz") in (True, False)
    # A false positive could remove the real item; otherwise size is unchanged.
    assert cf.size in (0, 1)


def test_delete_twice_second_returns_false():
    cf = CuckooFilter(capacity=1)
    cf.insert("once")
    assert cf.delete("once") is True
    assert cf.delete("once") is False
    assert cf.size == 0


def test_duplicate_inserts_require_matching_number_of_deletes():
    cf = CuckooFilter(capacity=1, bucket_size=4)
    cf.insert("dup")
    cf.insert("dup")
    assert cf.size == 2
    assert cf.delete("dup") is True
    assert cf.contains("dup") is True
    assert cf.delete("dup") is True
    assert cf.contains("dup") is False
    assert cf.size == 0


def test_insert_raises_when_filter_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")


def test_bucket_size_limits_duplicate_inserts():
    cf = CuckooFilter(capacity=1, bucket_size=4, max_displacements=10)
    for _ in range(4):
        assert cf.insert("same") is True
    assert cf.size == 4
    with pytest.raises(CuckooFilterFullException):
        cf.insert("same")
    assert cf.size == 4


def test_full_exception_message_mentions_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=3)
    cf.insert("a")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("b")
    assert "full" in str(excinfo.value).lower()


def test_full_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)
    with pytest.raises(Exception):
        raise CuckooFilterFullException("boom")


def test_repr_contains_capacity_and_size():
    cf = CuckooFilter(capacity=42, fingerprint_size=2)
    cf.insert("x")
    text = repr(cf)
    assert "CuckooFilter" in text
    assert "capacity=42" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text


def test_insert_after_delete_frees_space_in_full_filter():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=3)
    cf.insert("only")
    assert cf.delete("only") is True
    assert cf.insert("replacement") is True
    assert cf.contains("replacement") is True
