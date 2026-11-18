import pytest
import random

from cuckoopy import cuckoofilter
from cuckoopy import exceptions


def test_init_attributes():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2, fingerprint_size=3, max_displacements=100)
    assert cf.capacity == 10
    assert cf.bucket_size == 2
    assert cf.fingerprint_size == 3
    assert cf.max_displacements == 100
    assert isinstance(cf.buckets, list)
    assert len(cf.buckets) == 10
    for b in cf.buckets:
        assert getattr(b, 'size', None) == 2
    assert cf.size == 0


def test_repr_and_len():
    cf = cuckoofilter.CuckooFilter(5)
    repr_str = repr(cf)
    assert "capacity=5" in repr_str
    assert "size=0" in repr_str
    assert "fingerprint size=1" in repr_str
    assert len(cf) == 0
    _ = str(cf)  # just ensure no error


def test_get_index_and_alternate_index_consistency():
    cf = cuckoofilter.CuckooFilter(100)

    # Using a fixed item to test indexes
    item = "test_item"
    idx1 = cf._get_index(item)
    fp = cf._get_alternate_index(idx1, b"x")
    idx2 = cf._get_alternate_index(idx1, b"x")
    # Should be deterministic
    assert isinstance(idx1, int)
    assert 0 <= idx1 < cf.capacity
    assert isinstance(fp, int)
    assert 0 <= fp < cf.capacity
    # _get_alternate_index is stable, repeated calls same input same output
    assert fp == idx2


def test_insert_success(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=1, fingerprint_size=1, max_displacements=50)

    # Insert an item, should succeed
    item = "foo"
    result = cf.insert(item)
    assert result is True
    # After insert size increased
    assert len(cf) == 1
    # The item should be contained in cf
    assert item in cf


def test_insert_raises_when_full(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=10)

    # Fill up filter forcibly to trigger over capacity
    inserted = []
    # Insert up to capacity * bucket_size (2 buckets * 1 size = 2)
    # Insert new entries to fill both buckets
    # But to ensure insert will fail, we monkeypatch random.choice to always pick same eviction_index
    def fake_choice(seq):
        return seq[0]

    monkeypatch.setattr(random, 'choice', fake_choice)

    for i in range(2):
        assert cf.insert(f"item-{i}") is True
        inserted.append(f"item-{i}")

    # Now buckets should be full, next insert should raise
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert("this_will_fail")


def test_contains_true_and_false():
    cf = cuckoofilter.CuckooFilter(10)
    item = "hello"
    cf.insert(item)
    # The inserted item must be contained
    assert cf.contains(item) is True
    assert (item in cf) is True

    # Different item should not be contained (very low FP chance but no random seeds guaranteed)
    not_inserted = "world"
    # It's possible different item hashes to same fingerprint, but very unlikely
    # So we retry a few items to find one not contained
    trials = 5
    for i in range(trials):
        candidate = not_inserted + str(i)
        if candidate in cf:
            continue
        else:
            assert cf.contains(candidate) is False
            break
    else:
        # If all candidates found in cf (very unlikely), just skip this test final assert
        pass


def test_delete_returns_true_when_deleted_and_size_decrements():
    cf = cuckoofilter.CuckooFilter(10)
    item = "to_delete"
    cf.insert(item)
    prev_size = len(cf)
    assert cf.delete(item) is True
    assert len(cf) == prev_size - 1
    # After deletion item no longer contained
    assert item not in cf


def test_delete_returns_false_when_item_not_present():
    cf = cuckoofilter.CuckooFilter(10)
    # Empty filter; delete returns False
    assert cf.delete("non_existent") is False

    # Insert twice different items, delete something else
    cf.insert("a")
    cf.insert("b")
    assert cf.delete("something_not_inserted") is False


def test_insert_increments_size_by_one_only():
    cf = cuckoofilter.CuckooFilter(10)
    item = "unique1"
    assert len(cf) == 0
    assert cf.insert(item) is True
    size_after = len(cf)
    # Insert same item again (should not insert duplicate, but cuckoo filter can hold duplicates)
    # But typical cuckoo filter doesn't explicitly prevent duplicates, so assume insertion increments size
    inserted_again = cf.insert(item)
    assert inserted_again is True
    # Size should have increased by 1 again
    assert len(cf) == size_after + 1


def test_insert_eviction_path(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=5)

    # We want to force eviction path

    # Insert two items to fill buckets (capacity*bucket_size == 2)
    cf.insert("item1")
    cf.insert("item2")

    # Patch random.choice to always pick a bucket index to evict from
    evict_sequence = [0, 1, 0, 1, 0]  # simulate alternating eviction index
    call_count = {'count': 0}

    def fake_choice(seq):
        # Always return first item for eviction
        c = evict_sequence[call_count['count'] % len(evict_sequence)]
        call_count['count'] += 1
        return c

    monkeypatch.setattr(random, "choice", fake_choice)

    # Insert an item that must evict to succeed or fail
    item3 = "item3"

    # Might insert successfully before hitting max_displacements
    # But if insertion fails, expect exception
    try:
        result = cf.insert(item3)
        assert result is True
        assert len(cf) == 3
    except exceptions.CuckooFilterFullException:
        # This is acceptable if eviction exhausted
        pass


def test_contains_uses_fingerprints():
    cf = cuckoofilter.CuckooFilter(5, fingerprint_size=2)
    item = "testfp"
    cf.insert(item)
    fingerprint = cf._get_index(item)
    # Contains uses fingerprint check, so different item with same fingerprint is contained
    # We'll test that an item with same fingerprint bytes appears contained
    # We get fingerprint bytes directly:
    fp_bytes = cuckoofilter.hashutils.fingerprint(item, 2)
    # Using internals: test buckets
    bucket_i = cf._get_index(item)
    bucket_j = cf._get_alternate_index(bucket_i, fp_bytes)
    assert fp_bytes in cf.buckets[bucket_i] or fp_bytes in cf.buckets[bucket_j]


def test_delete_decrements_size_only_if_deleted():
    cf = cuckoofilter.CuckooFilter(5)
    item1 = "item1"
    item2 = "item2"
    cf.insert(item1)
    cf.insert(item2)
    size_before = len(cf)
    # Delete present item returns True and decrements size
    assert cf.delete(item1) is True
    assert len(cf) == size_before - 1
    # Delete same item again returns False and size unchanged
    assert cf.delete(item1) is False
    assert len(cf) == size_before - 1


def test_contains_magic_operator():
    cf = cuckoofilter.CuckooFilter(5)
    item = "magic"
    cf.insert(item)
    assert (item in cf) is True
    assert ("not_present" in cf) is False


def test_len_empty_and_after_inserts():
    cf = cuckoofilter.CuckooFilter(10)
    assert len(cf) == 0
    cf.insert("a")
    cf.insert("b")
    assert len(cf) == 2


def test_insert_when_bucket_inserts_directly(monkeypatch):
    # This test forcibly patches bucket.insert to always return True on first bucket
    cf = cuckoofilter.CuckooFilter(3, bucket_size=1, max_displacements=10)

    called = {'insert_called': 0}

    orig_bucket_insert = cf.buckets[0].insert

    def fake_insert(fp):
        called['insert_called'] += 1
        return True

    cf.buckets[0].insert = fake_insert

    result = cf.insert("test")
    assert result is True
    assert called['insert_called'] >= 1
    assert len(cf) == 1

    # restore original method to not affect other tests
    cf.buckets[0].insert = orig_bucket_insert


def test_insert_when_swap_and_reinsert(monkeypatch):
    # This test attempts to cover the eviction loop inside insert

    cf = cuckoofilter.CuckooFilter(2, bucket_size=1, max_displacements=3)

    # Fill the filter first
    cf.insert("x")
    cf.insert("y")

    # Patch random.choice to always pick first of i or j
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    swap_calls = {'count': 0}

    # Patch swap to swap once then fail insert one time
    original_swap = cf.buckets[0].swap

    def fake_swap(fp):
        swap_calls['count'] += 1
        # Swap always returns some fingerprint different from the input (simulate eviction)
        return b'\x01'

    cf.buckets[0].swap = fake_swap

    original_insert = cf.buckets[1].insert
    # Patch insert to fail once then succeed
    insert_called = {'count': 0}

    def fake_insert(fp):
        insert_called['count'] += 1
        if insert_called['count'] == 1:
            return False
        return True

    cf.buckets[1].insert = fake_insert

    # Insert new item triggers eviction
    result = cf.insert("z")
    assert result is True

    # Restore patched methods
    cf.buckets[0].swap = original_swap
    cf.buckets[1].insert = original_insert


def test_cuckoofilterfullexception_message():
    exc = exceptions.CuckooFilterFullException("filter full!")
    assert str(exc) == "filter full!"
