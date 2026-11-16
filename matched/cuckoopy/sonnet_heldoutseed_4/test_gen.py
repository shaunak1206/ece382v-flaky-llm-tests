import random

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture(autouse=True)
def _seed_random():
    random.seed(12345)


def test_init_stores_parameters():
    f = CuckooFilter(capacity=50, bucket_size=3, fingerprint_size=2,
                     max_displacements=10)
    assert f.capacity == 50
    assert f.bucket_size == 3
    assert f.fingerprint_size == 2
    assert f.max_displacements == 10
    assert len(f.buckets) == 50
    assert f.size == 0


def test_default_bucket_size_is_four():
    f = CuckooFilter(capacity=10)
    assert f.bucket_size == 4
    assert f.fingerprint_size == 1
    assert f.max_displacements == 500


def test_len_and_repr_on_empty_filter():
    f = CuckooFilter(capacity=10)
    assert len(f) == 0
    text = repr(f)
    assert "CuckooFilter" in text
    assert "capacity=10" in text
    assert "size=0" in text


def test_insert_returns_true_and_increments_size():
    f = CuckooFilter(capacity=100)
    assert f.insert("hello") is True
    assert len(f) == 1
    assert f.insert("world") is True
    assert len(f) == 2
    assert "size=2" in repr(f)


def test_contains_after_insert():
    f = CuckooFilter(capacity=100)
    f.insert("apple")
    assert f.contains("apple") is True


def test_contains_false_on_empty_filter():
    f = CuckooFilter(capacity=100)
    assert f.contains("missing") is False


def test_in_operator_delegates_to_contains():
    f = CuckooFilter(capacity=100)
    f.insert("banana")
    assert "banana" in f
    assert "cherry" not in f


def test_delete_existing_item():
    f = CuckooFilter(capacity=100)
    f.insert("item")
    assert f.delete("item") is True
    assert len(f) == 0
    assert f.contains("item") is False


def test_delete_missing_item_returns_false_and_keeps_size():
    f = CuckooFilter(capacity=100)
    f.insert("present")
    assert f.delete("absent-item-xyz") in (True, False)  # may collide on fingerprint
    f2 = CuckooFilter(capacity=100)
    assert f2.delete("anything") is False
    assert len(f2) == 0


def test_duplicate_inserts_require_matching_number_of_deletes():
    f = CuckooFilter(capacity=100, bucket_size=4)
    assert f.insert("dup")
    assert f.insert("dup")
    assert len(f) == 2
    assert f.delete("dup") is True
    assert f.contains("dup") is True
    assert f.delete("dup") is True
    assert f.contains("dup") is False
    assert len(f) == 0


def test_no_false_negatives_for_many_items():
    f = CuckooFilter(capacity=1000, bucket_size=4, fingerprint_size=2)
    items = ["item-%d" % i for i in range(200)]
    for item in items:
        assert f.insert(item) is True
    assert len(f) == 200
    for item in items:
        assert item in f


def test_delete_all_inserted_items_empties_filter():
    f = CuckooFilter(capacity=500, bucket_size=4, fingerprint_size=2)
    items = ["key-%d" % i for i in range(100)]
    for item in items:
        f.insert(item)
    for item in items:
        assert f.delete(item) is True
    assert len(f) == 0


def test_bucket_size_limits_entries_in_single_bucket():
    f = CuckooFilter(capacity=1, bucket_size=2, max_displacements=0)
    assert f.insert("a") is True
    assert f.insert("b") is True
    with pytest.raises(CuckooFilterFullException):
        f.insert("c")
    assert len(f) == 2


def test_full_exception_raised_with_zero_displacements():
    f = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    assert f.insert("first") is True
    with pytest.raises(CuckooFilterFullException) as excinfo:
        f.insert("second")
    assert "full" in str(excinfo.value).lower()
    assert len(f) == 1


def test_delete_frees_space_after_full():
    f = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    f.insert("first")
    with pytest.raises(CuckooFilterFullException):
        f.insert("second")
    assert f.delete("first") is True
    assert f.insert("second") is True
    assert f.contains("second") is True


def test_small_filter_eventually_raises_full_exception():
    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=20)
    with pytest.raises(CuckooFilterFullException):
        for i in range(50):
            f.insert("value-%d" % i)
    assert len(f) <= 2


def test_exception_is_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)
    with pytest.raises(CuckooFilterFullException, match="boom"):
        raise CuckooFilterFullException("boom")
