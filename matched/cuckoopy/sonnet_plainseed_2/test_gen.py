import pytest

from cuckoopy import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


# Note: string hashing is randomized per process, so tests that need exact
# bucket placement use capacity=1 (a single bucket) and the others only make
# assertions that hold for any placement. Wide fingerprints (4 bytes) are used
# when asserting absence to avoid fingerprint collisions.


def make_filter(capacity=1000, **kwargs):
    kwargs.setdefault("fingerprint_size", 4)
    return CuckooFilter(capacity=capacity, **kwargs)


def test_new_filter_is_empty():
    f = make_filter()
    assert len(f) == 0
    assert f.size == 0
    assert "anything" not in f


def test_bucket_size_default_is_four():
    f = CuckooFilter(capacity=10)
    assert f.bucket_size == 4
    assert all(b.size == 4 for b in f.buckets)


def test_bucket_size_custom_propagates_to_buckets():
    f = CuckooFilter(capacity=7, bucket_size=2)
    assert f.bucket_size == 2
    assert len(f.buckets) == 7
    assert all(b.size == 2 for b in f.buckets)


def test_insert_returns_true_and_updates_size():
    f = make_filter()
    assert f.insert("hello") is True
    assert len(f) == 1
    assert f.size == 1


def test_contains_after_insert():
    f = make_filter()
    f.insert("alpha")
    assert f.contains("alpha") is True
    assert "alpha" in f


def test_contains_returns_false_for_missing_item():
    f = make_filter()
    f.insert("alpha")
    assert f.contains("beta") is False
    assert "beta" not in f


def test_insert_multiple_distinct_items_all_found():
    f = make_filter(capacity=2000)
    items = ["item-%d" % i for i in range(50)]
    for item in items:
        assert f.insert(item) is True
    assert len(f) == len(items)
    for item in items:
        assert f.contains(item)


def test_delete_existing_item_returns_true_and_removes_it():
    f = make_filter()
    f.insert("gamma")
    assert f.delete("gamma") is True
    assert f.contains("gamma") is False
    assert len(f) == 0


def test_delete_missing_item_returns_false_and_keeps_size():
    f = make_filter()
    f.insert("gamma")
    assert f.delete("not-there") is False
    assert len(f) == 1
    assert f.contains("gamma")


def test_delete_on_empty_filter_returns_false():
    f = make_filter()
    assert f.delete("x") is False
    assert len(f) == 0


def test_delete_only_removes_one_copy_of_duplicate():
    f = make_filter(bucket_size=4)
    f.insert("dup")
    f.insert("dup")
    assert len(f) == 2
    assert f.delete("dup") is True
    assert len(f) == 1
    assert f.contains("dup")
    assert f.delete("dup") is True
    assert len(f) == 0
    assert not f.contains("dup")
    assert f.delete("dup") is False


def test_reinsert_after_delete():
    f = make_filter()
    f.insert("cycle")
    f.delete("cycle")
    assert f.insert("cycle") is True
    assert f.contains("cycle")
    assert len(f) == 1


def test_full_filter_raises_exception_on_single_bucket():
    f = CuckooFilter(capacity=1, bucket_size=2, fingerprint_size=4,
                     max_displacements=10)
    assert f.insert("one") is True
    assert f.insert("two") is True
    with pytest.raises(CuckooFilterFullException):
        f.insert("three")
    assert len(f) == 2


def test_bucket_size_controls_how_many_items_fit_in_single_bucket():
    f = CuckooFilter(capacity=1, bucket_size=3, fingerprint_size=4,
                     max_displacements=10)
    for name in ("a", "b", "c"):
        assert f.insert(name) is True
    assert len(f) == 3
    with pytest.raises(CuckooFilterFullException):
        f.insert("d")


def test_bucket_size_one_single_bucket_full_after_one_insert():
    f = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                     max_displacements=5)
    assert f.insert("first") is True
    with pytest.raises(CuckooFilterFullException):
        f.insert("second")
    assert len(f) == 1


def test_inserting_same_item_repeatedly_eventually_fills_filter():
    bucket_size = 2
    f = CuckooFilter(capacity=100, bucket_size=bucket_size,
                     fingerprint_size=4, max_displacements=20)
    with pytest.raises(CuckooFilterFullException):
        # An item can occupy at most 2 * bucket_size slots across its two
        # candidate buckets, so this must overflow regardless of hashing.
        for _ in range(2 * bucket_size + 1):
            f.insert("same")


def test_delete_frees_space_in_full_single_bucket():
    f = CuckooFilter(capacity=1, bucket_size=2, fingerprint_size=4,
                     max_displacements=10)
    f.insert("one")
    f.insert("two")
    assert f.delete("one") is True
    assert f.insert("three") is True
    assert len(f) == 2
    assert f.contains("two")
    assert f.contains("three")


def test_full_exception_is_exception_subclass_with_message():
    assert issubclass(CuckooFilterFullException, Exception)
    f = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=4,
                     max_displacements=3)
    f.insert("x")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        f.insert("y")
    assert "full" in str(excinfo.value).lower()


def test_exception_can_be_raised_and_caught_as_exception():
    with pytest.raises(Exception):
        raise CuckooFilterFullException("boom")


def test_len_and_repr_reflect_state():
    f = CuckooFilter(capacity=50, bucket_size=4, fingerprint_size=2)
    f.insert("r")
    text = repr(f)
    assert "capacity=50" in text
    assert "size=1" in text
    assert "fingerprint size=2" in text
    assert len(f) == 1
