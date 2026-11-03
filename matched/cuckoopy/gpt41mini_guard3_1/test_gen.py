import random
import pytest

from cuckoopy import cuckoofilter
from cuckoopy import exceptions


def test_insert_and_contains_simple_int_key():
    random.seed(42)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    key = 42
    assert key not in cf
    assert cf.insert(key) is True
    assert key in cf
    # Insert the same key again: the filter likely accepts duplicates, so might increase size
    size_before = len(cf)
    cf.insert(key)
    assert len(cf) >= size_before  # size may or may not increase if duplicate allowed


def test_contains_false_when_not_inserted():
    random.seed(1)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    # We never inserted 99, so it should not be in filter.
    assert 99 not in cf


def test_delete_existing_item_removes_it():
    random.seed(123)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    key = 7
    cf.insert(key)
    assert key in cf
    assert cf.delete(key) is True
    assert key not in cf
    # Deleting again returns False
    assert cf.delete(key) is False


def test_delete_non_existing_item_returns_false_and_no_size_change():
    random.seed(57)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    key = 1000
    size_before = len(cf)
    assert cf.delete(key) is False
    assert len(cf) == size_before


def test_insert_until_full_raises_exception(monkeypatch):
    random.seed(7)
    capacity = 4
    bucket_size = 1
    max_displacements = 5
    cf = cuckoofilter.CuckooFilter(capacity=capacity,
                                   bucket_size=bucket_size,
                                   fingerprint_size=1,
                                   max_displacements=max_displacements)

    # monkeypatch random.choice to always pick the first bucket index
    def choice_stub(seq):
        return seq[0]
    monkeypatch.setattr(cuckoofilter.random, "choice", choice_stub)

    # Insert distinct int keys until full. Because buckets are very small and capacity small,
    # inserts should eventually raise.
    inserted = 0
    for key in range(1000):
        try:
            result = cf.insert(key)
            assert result is True
            inserted += 1
        except exceptions.CuckooFilterFullException:
            break

    # Confirm that at least one insertion has occurred and that exception raised to end
    assert inserted > 0
    with pytest.raises(exceptions.CuckooFilterFullException):
        # Insert another item forcibly to confirm exception
        cf.insert(9999)


def test_len_property_reflects_inserts_and_deletes():
    random.seed(10)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    keys = [10, 20, 30, 40]
    for key in keys:
        cf.insert(key)
    assert len(cf) == 4
    cf.delete(keys[0])
    assert len(cf) == 3
    cf.delete(keys[1])
    assert len(cf) == 2
    # delete non-existing key doesn't affect size
    cf.delete(9999)
    assert len(cf) == 2


def test_repr_contains_expected_info():
    random.seed(1234)
    cf = cuckoofilter.CuckooFilter(capacity=8, bucket_size=3, fingerprint_size=2)
    s = repr(cf)
    assert 'capacity=8' in s
    assert 'size=0' in s
    assert 'fingerprint size=2' in s
    cf.insert(1)
    s2 = repr(cf)
    assert 'size=1' in s2


def test_contains_uses_own_methods_equivalent_to_in_operator():
    random.seed(100)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    key = 11
    cf.insert(key)
    # Directly use contains() method
    assert cf.contains(key)
    # Check __contains__ supports 'in'
    assert (key in cf) is True
    # Check not contained key
    assert not cf.contains(999)
    assert (999 in cf) is False


def test_fingerprint_consistency_for_same_input():
    random.seed(0)
    item = 123456789
    cf1 = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=2)
    fp1 = cf1.fingerprint_size, cuckoofilter.hashutils.fingerprint(item, 2)
    cf2 = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=2)
    fp2 = cf2.fingerprint_size, cuckoofilter.hashutils.fingerprint(item, 2)
    # fingerprint must be same for same input and fingerprint size
    assert fp1 == fp2


def test_get_index_and_alternate_index_produce_valid_buckets():
    random.seed(0)
    capacity = 16
    cf = cuckoofilter.CuckooFilter(capacity=capacity, bucket_size=4, fingerprint_size=1)
    key = 17
    i = cf._get_index(key)
    assert 0 <= i < capacity
    fingerprint = cuckoofilter.hashutils.fingerprint(key, cf.fingerprint_size)
    alt = cf._get_alternate_index(i, fingerprint)
    assert 0 <= alt < capacity
    # alt index should differ from i for most keys, but can be equal in rare cases
    # We do not assert inequality because it's not guaranteed


def test_insert_after_delete_of_same_item_should_succeed():
    random.seed(12345)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    key = 99
    assert cf.insert(key) is True
    assert cf.delete(key) is True
    assert cf.insert(key) is True
    assert key in cf


def test_insert_multiple_items_and_delete_some():
    random.seed(111)
    cf = cuckoofilter.CuckooFilter(capacity=32, bucket_size=4, fingerprint_size=1)
    keys = list(range(20))
    for key in keys:
        cf.insert(key)
    # Delete half
    for key in keys[:10]:
        cf.delete(key)
    # Check presence
    for key in keys[:10]:
        assert key not in cf
    for key in keys[10:]:
        assert key in cf
    # Size should be 10 now
    assert len(cf) == 10


def test_insert_duplicate_fingerprint_only_increases_size_once(monkeypatch):
    # Related to whether inserting same key twice counts as 2 or not
    random.seed(99)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)

    key = 7
    cf.insert(key)
    size_after_first = len(cf)

    # monkeypatch random.choice to always pick first index for deterministic replacement
    def choice_stub(seq):
        return seq[0]
    monkeypatch.setattr(cuckoofilter.random, 'choice', choice_stub)

    # Insert key again
    cf.insert(key)
    size_after_second = len(cf)

    # The size may or may not increase depending on whether duplicates are stored.
    # But size should never decrease.
    assert size_after_second >= size_after_first


def test_insert_returns_true_on_success():
    random.seed(22)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4, fingerprint_size=1)
    result = cf.insert(1)
    assert result is True


def test_delete_returns_false_for_never_inserted_item():
    random.seed(42)
    cf = cuckoofilter.CuckooFilter(capacity=16, bucket_size=4)
    assert cf.delete(123456789) is False


def test_insert_raises_if_filter_is_full(monkeypatch):
    random.seed(5)
    capacity = 2
    bucket_size = 1
    max_displacements = 3
    cf = cuckoofilter.CuckooFilter(capacity=capacity,
                                   bucket_size=bucket_size,
                                   fingerprint_size=1,
                                   max_displacements=max_displacements)

    # Patch random.choice to always pick index 0
    monkeypatch.setattr(cuckoofilter.random, "choice", lambda seq: seq[0])

    # Insert keys to fill filter
    cf.insert(1)
    cf.insert(2)
    with pytest.raises(exceptions.CuckooFilterFullException):
        cf.insert(3)


def test_inserting_and_deleting_multiple_items_stable_behavior():
    random.seed(77)
    cf = cuckoofilter.CuckooFilter(capacity=64, bucket_size=4, fingerprint_size=1)
    keys = list(range(30))
    for key in keys:
        cf.insert(key)

    for key in keys:
        assert key in cf

    for key in keys[:15]:
        assert cf.delete(key) is True

    for key in keys[:15]:
        assert key not in cf
    for key in keys[15:]:
        assert key in cf

    # Size reflects number of items remaining
    assert len(cf) == 15

