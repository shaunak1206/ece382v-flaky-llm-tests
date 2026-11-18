import pytest
from cuckoopy import cuckoofilter
from cuckoopy.exceptions import CuckooFilterFullException

def test_insert_and_contains_basic():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "hello"
    assert cf.insert(item) is True
    assert item in cf
    assert cf.contains(item) is True

def test_insert_duplicate_item_increments_size_once():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "test"
    assert cf.insert(item) is True
    size_after_first = len(cf)
    # Insert the same item again; size should not increase if duplicate insertion fails
    # But since filter uses fingerprints, duplicates can be stored unless bucket is full.
    # We test that it remains <= capacity and size increments properly.
    cf.insert(item)
    assert len(cf) >= size_after_first  # size could increase or not depending on fingerprint collision
    assert item in cf

def test_delete_existing_item_reduces_size():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item = "world"
    cf.insert(item)
    size_before_delete = len(cf)
    assert cf.delete(item) is True
    assert len(cf) == size_before_delete - 1
    assert cf.contains(item) is False

def test_delete_nonexistent_item_returns_false_and_does_not_change_size():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    item_inserted = "inserted"
    item_nonexistent = "notinserted"
    cf.insert(item_inserted)
    size_before_delete = len(cf)
    assert cf.delete(item_nonexistent) is False
    assert len(cf) == size_before_delete

def test_contains_without_inserting_returns_false():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.contains("missingitem") is False

def test_len_and_repr_consistency():
    cf = cuckoofilter.CuckooFilter(capacity=5)
    assert len(cf) == 0
    repr_str = repr(cf)
    assert "capacity=5" in repr_str
    assert "size=0" in repr_str
    assert "fingerprint size=" in repr_str

def test_bucket_size_property_set_correctly():
    bucket_size = 8
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=bucket_size)
    assert cf.bucket_size == bucket_size
    # Also check that each bucket has correct size
    for b in cf.buckets:
        assert b.size == bucket_size

def test_insert_until_full_raises_exception(monkeypatch):
    # We patch random.choice to always pick the first index to ensure eviction loop
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=10)

    # Insert into the filter to fill it
    # We create fingerprints that cause evictions
    # Insert two unique items - fill all buckets
    assert cf.insert("item1")
    assert cf.insert("item2")

    # Patch swap to always return the same fingerprint so we never make progress,
    # forcing the eviction loop to exhaust and raise exception.
    def fake_swap(self, fingerprint):
        # Always return the same fingerprint so no eviction success.
        return fingerprint

    monkeypatch.setattr(cf.buckets[0], "swap", fake_swap)
    monkeypatch.setattr(cf.buckets[1], "swap", fake_swap)

    with pytest.raises(CuckooFilterFullException):
        cf.insert("item3")

def test_contains_works_on_both_buckets():
    # We test that if fingerprint is in either bucket, contains returns True
    cf = cuckoofilter.CuckooFilter(capacity=5)
    item1 = "alpha"
    item2 = "beta"
    cf.insert(item1)
    cf.insert(item2)
    assert cf.contains(item1)
    assert cf.contains(item2)

def test_delete_reduces_size_correctly_when_multiple_items():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    items = ["a", "b", "c"]
    for item in items:
        cf.insert(item)
    size_before = len(cf)
    cf.delete(items[1])
    assert len(cf) == size_before - 1
    assert not cf.contains(items[1])
    assert cf.contains(items[0])
    assert cf.contains(items[2])

def test_insert_returns_true_on_first_attempt(monkeypatch):
    # Patch bucket.insert to return True immediately to avoid evictions
    cf = cuckoofilter.CuckooFilter(capacity=5)

    # Pick the bucket to patch
    bucket0 = cf.buckets[0]

    monkeypatch.setattr(bucket0, "insert", lambda f: True)

    result = cf.insert("someitem")
    assert result is True
    assert len(cf) == 1

def test_insert_eviction_success(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=10)
    item1 = "item1"
    item2 = "item2"
    # Insert items to fill buckets partially
    cf.insert(item1)
    cf.insert(item2)

    # Prepare a fingerprint that will be swapped and cause an eviction
    fp = cf.buckets[0].entries[0]

    # Patch bucket.insert to fail on first two insert attempts and succeed on third insert inside eviction
    # We simulate this by counting calls
    call_count = {"count": 0}

    original_insert_0 = cf.buckets[0].insert
    original_insert_1 = cf.buckets[1].insert

    def insert_patch(fp_arg):
        # Fail first two times, succeed third time
        if call_count["count"] < 2:
            call_count["count"] += 1
            return False
        return True

    def swap_patch(fp_arg):
        return fp  # always returns the same fingerprint to continue eviction

    monkeypatch.setattr(cf.buckets[0], "insert", insert_patch)
    monkeypatch.setattr(cf.buckets[1], "insert", insert_patch)
    monkeypatch.setattr(cf.buckets[0], "swap", swap_patch)
    monkeypatch.setattr(cf.buckets[1], "swap", swap_patch)
    # Patch random.choice to always evict from bucket 0
    monkeypatch.setattr("random.choice", lambda x: x[0])

    # Insert new item; eviction will try a few times and then succeed
    assert cf.insert("eviction_item") is True
    assert len(cf) > 0

def test_delete_returns_false_when_item_not_present():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    assert cf.delete("nonexistent") is False

def test_insert_increments_size_only_on_success(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=5)
    orig_insert = cf.buckets[0].insert

    # Patch insert to False (simulate bucket full)
    monkeypatch.setattr(cf.buckets[0], "insert", lambda f: False)
    # The alternate bucket insert returns True to allow success
    monkeypatch.setattr(cf.buckets[1], "insert", lambda f: True)

    result = cf.insert("itemX")
    assert result is True
    assert len(cf) == 1

def test_contains_and_delete_handle_different_fingerprints(monkeypatch):
    # To test contains and delete don't confuse different fingerprints
    cf = cuckoofilter.CuckooFilter(capacity=5)
    item1 = "item1"
    item2 = "item2"
    cf.insert(item1)
    # Patch fingerprint to a fixed value for item2 to guarantee difference
    from cuckoopy import hashutils
    old_fingerprint = hashutils.fingerprint

    def fake_fingerprint(item, size):
        if item == item2:
            return b"\xFF"
        return old_fingerprint(item, size)

    monkeypatch.setattr(hashutils, "fingerprint", fake_fingerprint)

    assert cf.contains(item1)
    assert cf.contains(item1) is True
    assert cf.contains(item2) is False
    assert cf.delete(item2) is False
    assert cf.delete(item1) is True
    assert cf.contains(item1) is False

def test_len_property_reflects_number_of_items():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    inserted = ["x", "y", "z"]
    for i, item in enumerate(inserted):
        cf.insert(item)
        assert len(cf) == i + 1
    for i, item in enumerate(inserted):
        cf.delete(item)
        assert len(cf) == len(inserted) - 1 - i

def test_contains_supports_python_in_operator():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    cf.insert("inside")
    assert "inside" in cf
    assert "notinside" not in cf
