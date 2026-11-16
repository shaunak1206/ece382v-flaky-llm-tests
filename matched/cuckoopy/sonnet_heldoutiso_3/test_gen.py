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


def test_init_custom_bucket_size():
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=2,
                      max_displacements=10)
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 10
    assert len(cf.buckets) == 8


def test_empty_filter_does_not_contain_item():
    cf = CuckooFilter(capacity=16)
    assert cf.contains("hello") is False
    assert len(cf) == 0


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=16)
    assert cf.insert("apple") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=16)
    cf.insert("apple")
    assert cf.contains("apple") is True


def test_in_operator_matches_contains():
    cf = CuckooFilter(capacity=16)
    cf.insert("banana")
    assert "banana" in cf
    assert cf.contains("banana") is True


def test_insert_multiple_items_all_found():
    cf = CuckooFilter(capacity=64)
    items = ["item-%d" % i for i in range(50)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 50
    for item in items:
        assert item in cf


def test_insert_many_items_with_displacements():
    # Fill the filter to a fairly high load so evictions are likely
    cf = CuckooFilter(capacity=100, bucket_size=4, fingerprint_size=2)
    items = ["key-%d" % i for i in range(250)]
    for item in items:
        cf.insert(item)
    assert len(cf) == 250
    for item in items:
        assert cf.contains(item)


def test_delete_existing_item():
    cf = CuckooFilter(capacity=16)
    cf.insert("apple")
    assert cf.delete("apple") is True
    assert len(cf) == 0
    assert cf.contains("apple") is False


def test_delete_missing_item_returns_false():
    cf = CuckooFilter(capacity=16)
    assert cf.delete("ghost") is False
    assert len(cf) == 0


def test_delete_twice_second_returns_false():
    cf = CuckooFilter(capacity=16)
    cf.insert("apple")
    assert cf.delete("apple") is True
    assert cf.delete("apple") is False
    assert cf.size == 0


def test_delete_only_removes_one_copy_of_duplicate():
    cf = CuckooFilter(capacity=16, bucket_size=4)
    cf.insert("dup")
    cf.insert("dup")
    assert len(cf) == 2
    assert cf.delete("dup") is True
    assert len(cf) == 1
    assert cf.contains("dup") is True
    assert cf.delete("dup") is True
    assert len(cf) == 0
    assert cf.contains("dup") is False


def test_delete_does_not_affect_other_items():
    cf = CuckooFilter(capacity=64)
    cf.insert("one")
    cf.insert("two")
    cf.delete("one")
    assert cf.contains("two") is True
    assert len(cf) == 1


def test_insert_raises_when_full_with_no_displacements():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    assert cf.insert("a") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("b")
    assert len(cf) == 1


def test_insert_raises_when_full_with_displacements():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=20)
    cf.insert("a")
    with pytest.raises(CuckooFilterFullException, match="full"):
        cf.insert("b")
    assert cf.size == 1


def test_filter_eventually_full_raises():
    cf = CuckooFilter(capacity=2, bucket_size=2, max_displacements=10)
    with pytest.raises(CuckooFilterFullException):
        for i in range(100):
            cf.insert("x-%d" % i)
    # at most capacity * bucket_size items can be stored
    assert len(cf) <= 4


def test_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)
    exc = CuckooFilterFullException("boom")
    assert str(exc) == "boom"


def test_repr_contains_state():
    cf = CuckooFilter(capacity=12, fingerprint_size=2)
    cf.insert("a")
    text = repr(cf)
    assert "CuckooFilter" in text
    assert "capacity=12" in text
    assert "size=1" in text
    assert "2 byte(s)" in text
