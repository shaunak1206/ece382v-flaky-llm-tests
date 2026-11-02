import pytest

from cuckoopy import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_defaults():
    cf = CuckooFilter(capacity=10)
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_bucket_size_attribute_is_stored():
    cf = CuckooFilter(capacity=8, bucket_size=2)
    assert cf.bucket_size == 2
    assert all(b.size == 2 for b in cf.buckets)


def test_bucket_size_custom_large():
    cf = CuckooFilter(capacity=5, bucket_size=7)
    assert cf.bucket_size == 7
    assert len(cf.buckets) == 5
    assert all(b.size == 7 for b in cf.buckets)


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100)
    assert cf.insert("apple") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=100)
    cf.insert("banana")
    assert cf.contains("banana") is True
    assert "banana" in cf


def test_contains_on_empty_filter_is_false():
    cf = CuckooFilter(capacity=100)
    assert cf.contains("missing") is False
    assert "missing" not in cf


def test_no_false_negatives_for_many_items():
    cf = CuckooFilter(capacity=1000, bucket_size=4, fingerprint_size=2)
    items = ["item-%d" % n for n in range(100)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 100
    for item in items:
        assert cf.contains(item) is True


def test_delete_existing_item():
    cf = CuckooFilter(capacity=100)
    cf.insert("cherry")
    assert cf.delete("cherry") is True
    assert cf.size == 0
    assert cf.contains("cherry") is False


def test_delete_missing_item_on_empty_filter_returns_false():
    cf = CuckooFilter(capacity=100)
    assert cf.delete("ghost") is False
    assert cf.size == 0


def test_delete_twice_second_returns_false():
    cf = CuckooFilter(capacity=100)
    cf.insert("durian")
    assert cf.delete("durian") is True
    assert cf.delete("durian") is False
    assert len(cf) == 0


def test_duplicate_insert_counts_twice_and_delete_removes_one():
    cf = CuckooFilter(capacity=100)
    assert cf.insert("fig") is True
    assert cf.insert("fig") is True
    assert cf.size == 2
    assert cf.delete("fig") is True
    assert cf.size == 1
    assert cf.contains("fig") is True
    assert cf.delete("fig") is True
    assert cf.contains("fig") is False


def test_reinsert_after_delete():
    cf = CuckooFilter(capacity=1, bucket_size=1)
    cf.insert("a")
    assert cf.delete("a") is True
    assert cf.insert("b") is True
    assert cf.contains("b") is True
    assert cf.size == 1


def test_insert_raises_when_filter_is_full():
    cf = CuckooFilter(capacity=1, bucket_size=1)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert cf.size == 1


def test_fills_exactly_bucket_size_then_raises():
    cf = CuckooFilter(capacity=1, bucket_size=3)
    for n in range(3):
        assert cf.insert("key-%d" % n) is True
    assert cf.size == 3
    with pytest.raises(CuckooFilterFullException):
        cf.insert("key-overflow")
    assert cf.size == 3


def test_full_exception_message_and_type():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    cf.insert("x")
    with pytest.raises(CuckooFilterFullException, match="Filter is full"):
        cf.insert("y")
    assert issubclass(CuckooFilterFullException, Exception)


def test_exception_can_be_raised_and_caught_directly():
    with pytest.raises(CuckooFilterFullException) as excinfo:
        raise CuckooFilterFullException("boom")
    assert str(excinfo.value) == "boom"


def test_len_tracks_inserts_and_deletes():
    cf = CuckooFilter(capacity=200, fingerprint_size=2)
    assert len(cf) == 0
    for n in range(10):
        cf.insert("k%d" % n)
    assert len(cf) == 10
    for n in range(4):
        assert cf.delete("k%d" % n) is True
    assert len(cf) == 6


def test_repr_contains_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=10, fingerprint_size=2)
    cf.insert("z")
    text = repr(cf)
    assert "capacity=10" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text


def test_unicode_items_roundtrip():
    cf = CuckooFilter(capacity=100, fingerprint_size=2)
    cf.insert("café")
    assert cf.contains("café") is True
    assert cf.delete("café") is True
    assert cf.contains("café") is False
