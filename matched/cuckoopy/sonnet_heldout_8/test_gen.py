import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_default_parameters():
    f = CuckooFilter(capacity=10)
    assert f.capacity == 10
    assert f.bucket_size == 4
    assert f.fingerprint_size == 1
    assert f.max_displacements == 500
    assert f.size == 0
    assert len(f.buckets) == 10


def test_bucket_size_custom_value():
    f = CuckooFilter(capacity=8, bucket_size=2)
    assert f.bucket_size == 2
    assert len(f.buckets) == 8


def test_bucket_size_one_allowed():
    f = CuckooFilter(capacity=8, bucket_size=1)
    assert f.bucket_size == 1
    assert f.insert("only") is True
    assert f.contains("only")


def test_insert_returns_true_and_increments_size():
    f = CuckooFilter(capacity=100)
    assert f.insert("hello") is True
    assert f.size == 1
    assert len(f) == 1


def test_contains_empty_filter_is_false():
    f = CuckooFilter(capacity=100)
    assert f.contains("anything") is False


def test_contains_after_insert():
    f = CuckooFilter(capacity=100)
    f.insert("hello")
    assert f.contains("hello") is True


def test_in_operator_matches_contains():
    f = CuckooFilter(capacity=100)
    f.insert("hello")
    assert "hello" in f
    assert ("hello" in f) == f.contains("hello")


def test_no_false_negatives_for_many_items():
    f = CuckooFilter(capacity=1000, bucket_size=4, fingerprint_size=2)
    items = ["item-{}".format(n) for n in range(200)]
    for item in items:
        assert f.insert(item) is True
    assert len(f) == 200
    for item in items:
        assert f.contains(item), item


def test_delete_existing_item():
    f = CuckooFilter(capacity=100)
    f.insert("hello")
    assert f.delete("hello") is True
    assert f.size == 0
    assert f.contains("hello") is False


def test_delete_missing_item_returns_false_and_keeps_size():
    f = CuckooFilter(capacity=100)
    assert f.delete("missing") is False
    assert f.size == 0

    f.insert("present")
    assert f.delete("missing-other-item-xyz") in (True, False)  # may collide
    # deleting from an empty filter again must be False
    g = CuckooFilter(capacity=100)
    assert g.delete("nothing") is False


def test_delete_only_removes_one_item_size_tracking():
    f = CuckooFilter(capacity=1000, fingerprint_size=2)
    for n in range(10):
        f.insert("k{}".format(n))
    assert f.size == 10
    assert f.delete("k3") is True
    assert f.size == 9
    assert f.delete("k4") is True
    assert f.size == 8


def test_reinsert_after_delete():
    f = CuckooFilter(capacity=100)
    f.insert("x")
    f.delete("x")
    assert f.insert("x") is True
    assert f.contains("x")
    assert f.size == 1


def test_insert_raises_when_filter_full():
    f = CuckooFilter(capacity=2, bucket_size=1, max_displacements=10)
    with pytest.raises(CuckooFilterFullException):
        for n in range(50):
            f.insert("full-{}".format(n))
    assert f.size <= f.capacity * f.bucket_size


def test_full_exception_message_and_type():
    f = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    f.insert("a")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        for n in range(20):
            f.insert("b{}".format(n))
    assert "full" in str(excinfo.value).lower()
    assert isinstance(excinfo.value, Exception)


def test_exception_can_be_raised_and_caught_directly():
    with pytest.raises(CuckooFilterFullException, match="boom"):
        raise CuckooFilterFullException("boom")
    assert issubclass(CuckooFilterFullException, Exception)


def test_repr_and_len():
    f = CuckooFilter(capacity=10, fingerprint_size=2)
    assert repr(f) == ('<CuckooFilter: capacity=10, size=0, '
                       'fingerprint size=2 byte(s)>')
    f.insert("a")
    assert len(f) == 1
    assert "size=1" in repr(f)
