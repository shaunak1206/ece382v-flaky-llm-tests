"""Unit tests for CuckooFilter and CuckooFilterFullException."""

import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


# ---------------------------------------------------------------------------
# Constructor and simple attributes
# ---------------------------------------------------------------------------

def test_init_stores_constructor_arguments():
    cf = CuckooFilter(capacity=16, bucket_size=3, fingerprint_size=2,
                      max_displacements=42)
    assert cf.capacity == 16
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 42


def test_init_default_values():
    cf = CuckooFilter(capacity=8)
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 500
    assert cf.size == 0


def test_init_creates_one_bucket_per_capacity_slot():
    cf = CuckooFilter(capacity=10)
    assert len(cf.buckets) == 10
    assert all(b is not None for b in cf.buckets)


@pytest.mark.parametrize("bucket_size", [1, 2, 4, 8])
def test_bucket_size_property_reflects_argument(bucket_size):
    cf = CuckooFilter(capacity=8, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size


def test_repr_contains_capacity_size_and_fingerprint_size():
    cf = CuckooFilter(capacity=32, fingerprint_size=2)
    text = repr(cf)
    assert "capacity=32" in text
    assert "size=0" in text
    assert "fingerprint size=2 byte(s)" in text


def test_len_is_zero_for_new_filter():
    assert len(CuckooFilter(capacity=8)) == 0


# ---------------------------------------------------------------------------
# contains / __contains__
# ---------------------------------------------------------------------------

def test_contains_on_empty_filter_returns_false():
    cf = CuckooFilter(capacity=8)
    assert cf.contains("not-inserted") is False


def test_dunder_contains_matches_contains_method():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("apple")
    assert ("apple" in cf) == cf.contains("apple")
    assert ("apple" in cf) is True


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_returns_true():
    cf = CuckooFilter(capacity=8)
    assert cf.insert("item") is True


def test_insert_increments_size_and_len():
    cf = CuckooFilter(capacity=8)
    cf.insert("a")
    assert cf.size == 1
    assert len(cf) == 1
    cf.insert("b")
    assert len(cf) == 2


def test_inserted_item_is_contained():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("hello")
    assert cf.contains("hello") is True


def test_many_distinct_items_are_all_contained():
    # Power-of-two capacity keeps the alternate-index mapping symmetric,
    # so every successfully inserted item must remain findable.
    cf = CuckooFilter(capacity=128, bucket_size=4, fingerprint_size=2)
    items = ["item-{}".format(i) for i in range(50)]
    for item in items:
        assert cf.insert(item) is True
    assert len(cf) == 50
    for item in items:
        assert cf.contains(item) is True


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_inserted_item_returns_true():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("doc")
    assert cf.delete("doc") is True


def test_delete_decrements_size():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("x")
    cf.insert("y")
    cf.delete("x")
    assert cf.size == 1
    assert len(cf) == 1


def test_delete_single_item_makes_it_absent():
    cf = CuckooFilter(capacity=64, fingerprint_size=2)
    cf.insert("gone")
    assert cf.delete("gone") is True
    assert cf.contains("gone") is False
    assert len(cf) == 0


def test_delete_not_inserted_item_on_empty_filter_returns_false():
    cf = CuckooFilter(capacity=8)
    assert cf.delete("missing") is False
    assert cf.size == 0


def test_delete_all_inserted_items_empties_filter():
    cf = CuckooFilter(capacity=128, bucket_size=4, fingerprint_size=2)
    items = ["key-{}".format(i) for i in range(40)]
    for item in items:
        cf.insert(item)
    for item in items:
        assert cf.delete(item) is True
    assert len(cf) == 0


# ---------------------------------------------------------------------------
# CuckooFilterFullException
# ---------------------------------------------------------------------------

def test_full_exception_is_an_exception_subclass():
    assert issubclass(CuckooFilterFullException, Exception)


def test_full_exception_can_be_raised_with_message():
    with pytest.raises(CuckooFilterFullException, match="full"):
        raise CuckooFilterFullException("Filter is full.")


def test_insert_raises_full_exception_when_filter_is_full():
    # One slot total: the second distinct insert cannot succeed.
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1,
                      max_displacements=5)
    raised = False
    for i in range(100):
        try:
            cf.insert("value-{}".format(i))
        except CuckooFilterFullException:
            raised = True
            break
    assert raised, "expected CuckooFilterFullException on a tiny filter"


def test_failed_insert_does_not_increase_size():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1,
                      max_displacements=5)
    cf.insert("first")
    assert cf.size == 1
    with pytest.raises(CuckooFilterFullException):
        cf.insert("second")
    assert cf.size == 1
    assert len(cf) == 1


def test_full_exception_message_mentions_full():
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1,
                      max_displacements=3)
    cf.insert("a")
    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("b")
    assert "full" in str(excinfo.value).lower()


@pytest.mark.parametrize("max_displacements", [0, 1, 10])
def test_zero_or_small_displacements_still_raise_when_full(max_displacements):
    cf = CuckooFilter(capacity=1, bucket_size=1, fingerprint_size=1,
                      max_displacements=max_displacements)
    cf.insert("one")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("two")
