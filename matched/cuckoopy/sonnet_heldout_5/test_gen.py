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
    cf = CuckooFilter(capacity=7, bucket_size=2, fingerprint_size=2,
                      max_displacements=20)
    assert cf.capacity == 7
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 20
    assert len(cf.buckets) == 7


def test_empty_filter_does_not_contain_item():
    cf = CuckooFilter(capacity=100)
    assert cf.contains("hello") is False
    assert len(cf) == 0


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100)
    assert cf.insert("hello") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=100)
    cf.insert("apple")
    assert cf.contains("apple") is True


def test_dunder_contains_matches_contains():
    cf = CuckooFilter(capacity=100)
    cf.insert("apple")
    assert "apple" in cf
    assert cf.contains("apple") == ("apple" in cf)


def test_no_false_negatives_for_many_items():
    cf = CuckooFilter(capacity=1000, fingerprint_size=2)
    items = ["item-%d" % i for i in range(200)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 200
    for item in items:
        assert item in cf


def test_delete_existing_item():
    cf = CuckooFilter(capacity=100)
    cf.insert("banana")
    assert cf.delete("banana") is True
    assert cf.size == 0
    assert cf.contains("banana") is False


def test_delete_nonexistent_item_returns_false():
    cf = CuckooFilter(capacity=100)
    assert cf.delete("ghost") is False
    assert cf.size == 0


def test_delete_nonexistent_does_not_change_size():
    cf = CuckooFilter(capacity=100)
    cf.insert("present")
    assert cf.delete("ghost-item-never-inserted-xyz") in (True, False)
    # size is consistent with the result of the delete
    assert cf.size in (0, 1)


def test_delete_twice_second_returns_false():
    cf = CuckooFilter(capacity=100)
    cf.insert("once")
    assert cf.delete("once") is True
    assert cf.delete("once") is False
    assert cf.size == 0


def test_insert_duplicate_items_requires_multiple_deletes():
    cf = CuckooFilter(capacity=1000, bucket_size=4)
    cf.insert("dup")
    cf.insert("dup")
    assert cf.size == 2
    assert cf.delete("dup") is True
    assert cf.size == 1
    assert cf.contains("dup") is True
    assert cf.delete("dup") is True
    assert cf.size == 0
    assert cf.contains("dup") is False


def test_delete_all_inserted_items_empties_filter():
    cf = CuckooFilter(capacity=500, fingerprint_size=2)
    items = ["key-%d" % i for i in range(100)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0


def test_insert_raises_when_filter_full_single_slot():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=10)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert cf.size == 1


def test_insert_raises_when_small_filter_overfilled():
    cf = CuckooFilter(capacity=2, bucket_size=2, max_displacements=20)
    inserted = 0
    with pytest.raises(CuckooFilterFullException):
        for i in range(50):
            cf.insert("overflow-%d" % i)
            inserted += 1
    assert inserted <= cf.capacity * cf.bucket_size
    assert cf.size == inserted


def test_full_exception_message_and_type():
    assert issubclass(CuckooFilterFullException, Exception)
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    cf.insert("a")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("b")
    assert "full" in str(excinfo.value).lower()


def test_bucket_size_limits_capacity_for_identical_items():
    # Identical items map to the same two buckets, so at most 2 * bucket_size
    # copies can ever be stored (a single bucket if both indices coincide).
    cf = CuckooFilter(capacity=8, bucket_size=2, max_displacements=10)
    count = 0
    with pytest.raises(CuckooFilterFullException):
        for _ in range(100):
            cf.insert("same")
            count += 1
    assert 2 <= count <= 2 * cf.bucket_size


def test_repr_contains_parameters():
    cf = CuckooFilter(capacity=42, fingerprint_size=2)
    cf.insert("x")
    text = repr(cf)
    assert "CuckooFilter" in text
    assert "capacity=42" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text
