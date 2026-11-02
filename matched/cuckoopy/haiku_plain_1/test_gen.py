"""Unit tests for CuckooFilter and CuckooFilterFullException."""

import pytest

from cuckoopy import CuckooFilter, bucket, exceptions


def test_default_parameters():
    f = CuckooFilter(8)
    assert f.capacity == 8
    assert f.bucket_size == 4
    assert f.fingerprint_size == 1
    assert f.max_displacements == 500
    assert len(f) == 0


def test_constructor_stores_explicit_parameters():
    f = CuckooFilter(20, bucket_size=2, fingerprint_size=3,
                     max_displacements=42)
    assert f.capacity == 20
    assert f.bucket_size == 2
    assert f.fingerprint_size == 3
    assert f.max_displacements == 42


def test_constructor_creates_one_empty_bucket_per_slot():
    f = CuckooFilter(7)
    assert len(f.buckets) == 7
    for b in f.buckets:
        assert isinstance(b, bucket.Bucket)
        assert len(b) == 0


def test_bucket_size_is_applied_to_every_bucket():
    f = CuckooFilter(5, bucket_size=3)
    assert f.bucket_size == 3
    assert all(b.size == 3 for b in f.buckets)


def test_empty_filter_contains_nothing():
    f = CuckooFilter(10)
    for item in ["a", "apple", "hello world"]:
        assert item not in f
        assert f.contains(item) is False


def test_insert_returns_true_and_adds_item():
    f = CuckooFilter(10)
    assert f.insert("apple") is True
    assert len(f) == 1
    assert f.contains("apple") is True


def test_insert_several_distinct_items_are_all_found():
    f = CuckooFilter(100, fingerprint_size=4)
    items = ["alpha", "beta", "gamma", "delta"]
    for item in items:
        assert f.insert(item) is True
    assert len(f) == len(items)
    for item in items:
        assert f.contains(item)


def test_duplicate_insert_is_counted_twice():
    f = CuckooFilter(10)
    f.insert("dup")
    f.insert("dup")
    assert len(f) == 2
    assert f.contains("dup")


def test_delete_removes_one_copy_per_call():
    f = CuckooFilter(10)
    f.insert("dup")
    f.insert("dup")
    assert f.delete("dup") is True
    assert len(f) == 1
    assert f.contains("dup")
    assert f.delete("dup") is True
    assert len(f) == 0
    assert not f.contains("dup")
    assert f.delete("dup") is False
    assert len(f) == 0


def test_delete_single_item_makes_it_absent():
    f = CuckooFilter(10)
    f.insert("solo")
    assert f.delete("solo") is True
    assert "solo" not in f
    assert len(f) == 0


def test_delete_on_empty_filter_returns_false():
    f = CuckooFilter(10)
    assert f.delete("ghost") is False
    assert len(f) == 0


def test_delete_keeps_other_items_findable():
    f = CuckooFilter(100, fingerprint_size=4)
    for item in ["a", "b", "c"]:
        f.insert(item)
    assert f.delete("b") is True
    assert f.contains("a")
    assert f.contains("c")
    assert not f.contains("b")
    assert len(f) == 2


def test_len_tracks_mixed_inserts_and_deletes():
    f = CuckooFilter(100, fingerprint_size=4)
    f.insert("x")
    f.insert("y")
    assert len(f) == 2
    assert f.delete("x")
    assert len(f) == 1
    assert f.delete("x") is False
    assert len(f) == 1
    f.insert("z")
    assert len(f) == 2
    assert f.delete("y")
    assert len(f) == 1
    assert f.contains("z")


def test_in_operator_matches_contains_method():
    f = CuckooFilter(10, fingerprint_size=4)
    f.insert("present")
    assert ("present" in f) is True
    assert f.contains("present") is True
    assert ("absent" in f) is False
    assert f.contains("absent") is False


def test_insert_into_full_filter_raises():
    f = CuckooFilter(1, bucket_size=1, max_displacements=0)
    assert f.insert("a") is True
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("b")


def test_failed_insert_leaves_size_and_existing_item_unchanged():
    f = CuckooFilter(1, bucket_size=1, max_displacements=0)
    f.insert("a")
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("b")
    assert len(f) == 1
    assert f.contains("a")
    assert len(f.buckets[0]) == 1


def test_bucket_size_limits_entries_per_bucket():
    f = CuckooFilter(1, bucket_size=3, max_displacements=0)
    for item in ["a", "b", "c"]:
        assert f.insert(item) is True
    assert len(f.buckets[0]) == 3
    with pytest.raises(exceptions.CuckooFilterFullException):
        f.insert("d")
    assert len(f) == 3
    assert len(f.buckets[0]) == 3


def test_exception_is_an_exception_subclass():
    assert issubclass(exceptions.CuckooFilterFullException, Exception)
    with pytest.raises(Exception):
        raise exceptions.CuckooFilterFullException("boom")


def test_full_insert_error_message_mentions_full():
    f = CuckooFilter(1, bucket_size=1, max_displacements=0)
    f.insert("a")
    with pytest.raises(exceptions.CuckooFilterFullException) as excinfo:
        f.insert("b")
    assert "full" in str(excinfo.value)


def test_repr_reports_capacity_size_and_fingerprint_size():
    f = CuckooFilter(10, fingerprint_size=2)
    f.insert("a")
    text = repr(f)
    assert "capacity=10" in text
    assert "size=1" in text
    assert "fingerprint size=2 byte(s)" in text


def test_empty_string_is_a_valid_item():
    f = CuckooFilter(10)
    assert f.insert("") is True
    assert "" in f
    assert len(f) == 1


def test_unicode_item_round_trip():
    f = CuckooFilter(10, fingerprint_size=4)
    item = "naïve ✓"
    f.insert(item)
    assert item in f
    assert f.delete(item) is True
    assert item not in f


def test_deleting_every_item_empties_filter():
    f = CuckooFilter(50, fingerprint_size=4)
    items = ["one", "two", "three", "four"]
    for item in items:
        f.insert(item)
    for item in items:
        assert f.delete(item) is True
    assert len(f) == 0
    for item in items:
        assert item not in f
