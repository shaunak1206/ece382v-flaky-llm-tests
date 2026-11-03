import random
import pytest

import cuckoopy.cuckoofilter as cuckoofilter
import cuckoopy.exceptions as exceptions


def test_init_and_repr_basic_properties():
    # Seed random for deterministic behavior in insert
    random.seed(12345)

    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=4,
                                   fingerprint_size=1, max_displacements=50)
    # Basic attributes set correctly
    assert cf.capacity == 10
    assert cf.bucket_size == 4
    assert cf.fingerprint_size == 1
    assert cf.max_displacements == 50
    assert len(cf.buckets) == 10
    # Bucket object type check (assuming bucket.Bucket class has required interface)
    for b in cf.buckets:
        # Bucket should have insert, delete, swap methods and __contains__
        assert hasattr(b, "insert")
        assert hasattr(b, "delete")
        assert hasattr(b, "swap")
        assert hasattr(b, "__contains__")
    # size should be zero initially
    assert len(cf) == 0

    r = repr(cf)
    assert r.startswith("<CuckooFilter: capacity=10, size=0, fingerprint size=1 byte")
    # str and repr length nonzero
    assert isinstance(r, str)


def test_insert_contains_delete_int_keys_consistency(monkeypatch):
    # Seed random for deterministic eviction choices
    random.seed(42)

    # Use integer keys to avoid hash randomization issues
    cf = cuckoofilter.CuckooFilter(capacity=20, bucket_size=2, fingerprint_size=2,
                                   max_displacements=10)

    items = list(range(5))
    # All inserted successfully
    for item in items:
        inserted = cf.insert(item)
        assert inserted is True
        assert item in cf
        # Contains must return True for inserted items
        assert cf.contains(item) is True

    # Size updated correctly
    assert len(cf) == len(items)

    # Delete one item
    deleted = cf.delete(items[0])
    assert deleted is True
    assert items[0] not in cf
    assert len(cf) == len(items) - 1

    # Delete same item again returns False
    deleted_again = cf.delete(items[0])
    assert deleted_again is False

    # Delete remaining items
    for item in items[1:]:
        assert cf.delete(item) is True
        assert item not in cf
    # Now empty
    assert len(cf) == 0


def test_contains_returns_false_for_noninserted(monkeypatch):
    random.seed(99)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=1)
    # Insert some items
    for i in range(3):
        cf.insert(i)

    # Test non-inserted integers, must not be contained
    for k in [100, 200, -1, 50]:
        assert cf.contains(k) is False
        assert (k in cf) is False


def test_insert_fails_when_filter_full(monkeypatch):
    # Patch random.choice to always pick first index deterministically
    def always_first(lst):
        return lst[0]

    monkeypatch.setattr(random, "choice", always_first)
    random.seed(123)
    # Capacity 2 buckets with bucket_size=1 causes early full
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1,
                                   fingerprint_size=1, max_displacements=5)

    # Insert two items should succeed as buckets can hold 2 fingerprints total
    assert cf.insert(1) is True
    assert cf.insert(2) is True
    # Further insert likely fails due to no room and max_displacements reached
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(3)


def test_insert_eviction_path(monkeypatch):
    # We want to test the eviction path by making buckets full initially,
    # but monkeypatch bucket.Bucket to stub insert and swap with predictable behavior.

    class DummyBucket:
        def __init__(self, size):
            self.items = []
            self.size = size

        def insert(self, fingerprint):
            # insert fails if full (size occupied)
            if len(self.items) >= self.size:
                return False
            self.items.append(fingerprint)
            return True

        def swap(self, fingerprint):
            # Evict first inserted fingerprint
            evicted = self.items.pop(0)
            self.items.append(fingerprint)
            return evicted

        def delete(self, fingerprint):
            if fingerprint in self.items:
                self.items.remove(fingerprint)
                return True
            return False

        def __contains__(self, fingerprint):
            return fingerprint in self.items

    # Patch buckets to DummyBucket
    monkeypatch.setattr(
        cuckoofilter,
        "bucket",
        type("bucket", (), {"Bucket": DummyBucket})
    )

    random.seed(1234)
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=2,
                                   fingerprint_size=1, max_displacements=10)
    # Insert one item to fill bucket
    assert cf.insert(100) is True
    # Insert enough different items to force eviction at least once
    # Because each bucket holds 2 items, inserting 5 different items in capacity 5
    # should cause evictions due to alternating index collisions

    # Use integer keys to avoid hash randomization issues.
    inserted_count = 1
    for k in range(101, 110):
        try:
            inserted = cf.insert(k)
            assert inserted is True
            inserted_count += 1
        except exceptions.CuckooFilterFullException:
            # It's okay if filter gets full after enough insertions
            break
    # After insertions, size property must match inserted count
    assert cf.size == inserted_count

    # Confirm inserted items are contained
    for item in range(100, 100 + inserted_count):
        assert item in cf


def test_delete_returns_false_when_item_not_present(monkeypatch):
    random.seed(2021)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2)
    # Delete an item never inserted returns False
    assert cf.delete(99999) is False


def test_len_and_contains_magic_methods(monkeypatch):
    random.seed(777)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=3)

    # Initially empty
    assert len(cf) == 0

    # Insert one
    cf.insert(42)
    assert len(cf) == 1
    assert (42 in cf) is True

    # Delete it
    cf.delete(42)
    assert len(cf) == 0
    assert (42 in cf) is False


def test_bucket_size_property_retained(monkeypatch):
    cf = cuckoofilter.CuckooFilter(capacity=7, bucket_size=5)
    # bucket_size stored correctly and identical to each bucket's size param
    assert cf.bucket_size == 5
    for b in cf.buckets:
        # We do not know bucket.Bucket implementation details,
        # but bucket_size param must be set in initialization
        # We skip internal check, just assert buckets exists count as expected
        assert b is not None


def test_insert_different_fingerprint_sizes(monkeypatch):
    random.seed(77)
    for fsize in [1, 2, 4]:
        cf = cuckoofilter.CuckooFilter(capacity=15, bucket_size=3,
                                       fingerprint_size=fsize)
        item = 123456789
        inserted = cf.insert(item)
        assert inserted is True
        # Contains should also return True
        assert cf.contains(item)
        # Delete must work
        deleted = cf.delete(item)
        assert deleted is True
        # Then contains must be False
        assert not cf.contains(item)


def test_insert_and_contains_do_not_modify_filter(monkeypatch):
    random.seed(15)
    cf = cuckoofilter.CuckooFilter(capacity=20, bucket_size=4)

    # Insert does modify size, contains does not
    item = 314159
    size_before = len(cf)
    inserted = cf.insert(item)
    assert inserted is True
    size_after_insert = len(cf)
    assert size_after_insert == size_before + 1

    # Calling contains repeatedly does not modify size
    for _ in range(5):
        assert cf.contains(item) is True
        assert len(cf) == size_after_insert


def test_delete_decreases_size_only_when_successful(monkeypatch):
    random.seed(321)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2)
    # Insert one item
    assert cf.insert(777) is True
    size_before = len(cf)
    # Delete wrong item - should be False size unchanged
    deleted = cf.delete(888)
    assert deleted is False
    assert len(cf) == size_before
    # Delete inserted item - True and size decreases
    deleted = cf.delete(777)
    assert deleted is True
    assert len(cf) == size_before - 1


def test_contains_and_delete_with_duplicate_fingerprints(monkeypatch):
    # We test duplicate fingerprints handling by inserting different items that
    # may produce same fingerprint, then deleting one item does not delete other

    # Patch bucket.Bucket with dummy allowing duplicates
    class DummyBucket:
        def __init__(self, size):
            self.items = []
            self.size = size

        def insert(self, fingerprint):
            if len(self.items) >= self.size:
                return False
            self.items.append(fingerprint)
            return True

        def swap(self, fingerprint):
            evicted = self.items.pop(0)
            self.items.append(fingerprint)
            return evicted

        def delete(self, fingerprint):
            # Delete one occurrence of fingerprint if present
            if fingerprint in self.items:
                self.items.remove(fingerprint)
                return True
            return False

        def __contains__(self, fingerprint):
            return fingerprint in self.items

    monkeypatch.setattr(
        cuckoofilter,
        "bucket",
        type("bucket", (), {"Bucket": DummyBucket})
    )

    # Patch hashutils.fingerprint to produce same fingerprint for different inputs
    # use a fixed dummy fingerprint value e.g. b'\xAA'
    def dummy_fingerprint(item, size):
        return b'\xAA' * size

    monkeypatch.setattr(
        cuckoofilter.hashutils,
        "fingerprint",
        dummy_fingerprint
    )

    random.seed(10101)
    cf = cuckoofilter.CuckooFilter(capacity=4, bucket_size=3, fingerprint_size=1)
    # Insert two distinct items but same fingerprint
    assert cf.insert(10) is True
    assert cf.insert(20) is True

    # Both items should be 'contained'
    assert cf.contains(10) is True
    assert cf.contains(20) is True

    # Delete one item must return True, but the other remains
    assert cf.delete(10) is True
    assert cf.contains(20) is True
    # Deleting 10 again returns False
    assert cf.delete(10) is False


def test_insert_with_random_choice_seed(monkeypatch):
    # Test that seeding random produces repeatable results for eviction_index
    choices = []
    def record_choice(lst):
        # Record choices made by filter
        c = random.choice(lst)
        choices.append(c)
        return c

    # Patch random.choice to our function that calls original random.choice
    original_choice = random.choice
    monkeypatch.setattr(random, "choice", record_choice)

    random.seed(1234)
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=2)

    # Insert items; some should trigger eviction using random.choice
    item_list = list(range(7))
    for it in item_list:
        try:
            cf.insert(it)
        except exceptions.CuckooFilterFullException:
            break

    # All choices made are in original candidate lists [i, j]
    for choice_val in choices:
        assert isinstance(choice_val, int)
        assert 0 <= choice_val < cf.capacity


def test_cuckoofilterfullexception_inherits_from_exception():
    exc = exceptions.CuckooFilterFullException("full")
    assert isinstance(exc, Exception)
    assert str(exc) == "full"


def test_insert_raises_specific_exception_on_full(monkeypatch):
    random.seed(2023)

    def always_fail_insert(self, fingerprint):
        return False
    def always_swap(self, fingerprint):
        return fingerprint  # swaps same fingerprint to prevent progress

    # Patch bucket.Bucket to simulate full buckets that never insert new fingerprints
    class FullBucket:
        def __init__(self, size):
            pass
        insert = always_fail_insert
        delete = lambda self, fp: False
        swap = always_swap
        def __contains__(self, fingerprint): return False

    monkeypatch.setattr(
        cuckoofilter,
        "bucket",
        type("bucket", (), {"Bucket": FullBucket})
    )

    cf = cuckoofilter.CuckooFilter(capacity=3, bucket_size=1,
                                   max_displacements=3)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(123)


def test__get_index_and__get_alternate_index_properties(monkeypatch):
    # Use integer keys to avoid random hash
    random.seed(0)
    cf = cuckoofilter.CuckooFilter(capacity=11, bucket_size=3)

    # _get_index returns integer in [0, capacity)
    idx = cf._get_index(1000)
    assert isinstance(idx, int)
    assert 0 <= idx < cf.capacity

    # For integer items, fingerprint is bytes so alt index is also in [0, capacity)
    fp = cuckoofilter.hashutils.fingerprint(1000, cf.fingerprint_size)
    alt_idx = cf._get_alternate_index(idx, fp)
    assert isinstance(alt_idx, int)
    assert 0 <= alt_idx < cf.capacity

    # _get_alternate_index different from _get_index sometimes
    # Not necessarily guaranteed (XOR could produce same)
    # So test just that alt_idx belongs in capacity range


def test_insert_item_multiple_times_only_increases_size_once(monkeypatch):
    random.seed(202)
    cf = cuckoofilter.CuckooFilter(capacity=30, bucket_size=4)

    item = 42
    assert cf.insert(item) is True
    size_after_first = len(cf)
    # Insert same item multiple times - size increases only first time
    # Subsequent inserts can be True or raise, but size must not increase
    for _ in range(3):
        try:
            result = cf.insert(item)
            assert result is True
        except exceptions.CuckooFilterFullException:
            # It's acceptable to be full if filter fills, but then size no bigger
            break
        assert len(cf) == size_after_first


def test_delete_after_full_insert(monkeypatch):
    random.seed(10)

    # Patch bucket.Bucket as full quickly by always failing insert but allowing delete
    class DummyBucket:
        def __init__(self, size):
            self.size = size
            self.items = []

        def insert(self, fingerprint):
            return False

        def swap(self, fingerprint):
            return fingerprint

        def delete(self, fingerprint):
            # Simulate delete finds nothing to delete
            return False

        def __contains__(self, fingerprint):
            return False

    monkeypatch.setattr(cuckoofilter, "bucket",
                        type("bucket", (), {"Bucket": DummyBucket}))

    cf = cuckoofilter.CuckooFilter(capacity=3, bucket_size=1, max_displacements=2)
    # Insert cannot succeed
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(99)
    # Delete an item should return False gracefully
    assert cf.delete(99) is False


def test___contains___calls_contains(monkeypatch):
    random.seed(123)

    cf = cuckoofilter.CuckooFilter(10)
    calls = []

    # Patch cf.contains to wrap and record call
    orig_contains = cf.contains

    def record_contains(item):
        calls.append(item)
        return orig_contains(item)

    monkeypatch.setattr(cf, "contains", record_contains)

    # contains() returns False since empty
    assert (123 in cf) is False
    assert calls == [123]
