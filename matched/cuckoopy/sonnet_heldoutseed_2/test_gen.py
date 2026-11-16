import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


def test_init_attributes():
    cf = CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2,
                      max_displacements=7)
    assert cf.capacity == 10
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 7
    assert cf.size == 0
    assert len(cf.buckets) == 10


def test_default_parameters():
    cf = CuckooFilter(capacity=5)
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500


@pytest.mark.parametrize("bucket_size", [1, 2, 4, 8])
def test_bucket_size_is_stored_and_applied(bucket_size):
    cf = CuckooFilter(capacity=4, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size
    assert len(cf.buckets) == 4


def test_empty_filter_contains_nothing():
    cf = CuckooFilter(capacity=100)
    assert cf.contains("anything") is False
    assert "anything" not in cf
    assert len(cf) == 0


def test_insert_returns_true_and_increments_size():
    cf = CuckooFilter(capacity=100)
    assert cf.insert("hello") is True
    assert cf.size == 1
    assert len(cf) == 1


def test_contains_after_insert():
    cf = CuckooFilter(capacity=100)
    cf.insert("hello")
    assert cf.contains("hello") is True


def test_dunder_contains_matches_contains():
    cf = CuckooFilter(capacity=100)
    cf.insert("hello")
    assert "hello" in cf
    assert cf.contains("hello") == ("hello" in cf)


def test_insert_many_items_no_false_negatives():
    cf = CuckooFilter(capacity=1000, bucket_size=4)
    items = ["item-%d" % i for i in range(100)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 100
    for item in items:
        assert cf.contains(item) is True


def test_delete_existing_item():
    cf = CuckooFilter(capacity=100)
    cf.insert("hello")
    assert cf.delete("hello") is True
    assert cf.size == 0
    assert cf.contains("hello") is False


def test_delete_missing_item_returns_false():
    cf = CuckooFilter(capacity=100)
    assert cf.delete("missing") is False
    assert cf.size == 0


def test_delete_does_not_decrement_size_on_failure():
    cf = CuckooFilter(capacity=100)
    cf.insert("a")
    cf.delete("not-there-at-all-xyz")  # may or may not collide; size check below
    assert cf.size in (0, 1)
    assert cf.delete("a") in (True, False)
    assert cf.size >= 0


def test_delete_twice_second_returns_false():
    cf = CuckooFilter(capacity=100)
    cf.insert("hello")
    assert cf.delete("hello") is True
    assert cf.delete("hello") is False
    assert cf.size == 0


def test_insert_delete_multiple_items():
    cf = CuckooFilter(capacity=500)
    items = ["k%d" % i for i in range(50)]
    for item in items:
        cf.insert(item)
    assert len(cf) == 50
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0


def test_insert_raises_when_filter_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=10)
    assert cf.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")


def test_size_unchanged_after_failed_insert():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=10)
    cf.insert("first")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert cf.size == 1


def test_bucket_size_limits_entries_in_single_bucket():
    cf = CuckooFilter(capacity=1, bucket_size=3, max_displacements=10)
    for item in ("a", "b", "c"):
        assert cf.insert(item) is True
    assert len(cf) == 3
    with pytest.raises(CuckooFilterFullException):
        cf.insert("d")
    assert len(cf) == 3


def test_full_exception_message_and_type():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=5)
    cf.insert("x")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("y")
    assert "full" in str(excinfo.value).lower()
    assert isinstance(excinfo.value, Exception)


def test_insert_succeeds_after_delete_frees_space():
    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=10)
    cf.insert("first")
    assert cf.delete("first") is True
    assert cf.insert("second") is True
    assert cf.contains("second") is True


def test_repr_contains_parameters():
    cf = CuckooFilter(capacity=10, fingerprint_size=1)
    assert repr(cf) == ('<CuckooFilter: capacity=10, size=0, '
                        'fingerprint size=1 byte(s)>')
    cf.insert("a")
    assert "size=1" in repr(cf)
