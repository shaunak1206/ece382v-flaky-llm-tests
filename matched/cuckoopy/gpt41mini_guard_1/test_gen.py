import random
import pytest

from cuckoopy import cuckoofilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture(autouse=True)
def reset_random_seed():
    # Ensure random.seed is reset before each test to avoid flakiness
    random.seed(12345)
    yield
    random.seed(12345)


def test_init_properties():
    cf = cuckoofilter.CuckooFilter(capacity=10, bucket_size=3, fingerprint_size=2, max_displacements=100)
    assert cf.capacity == 10
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 100
    assert len(cf.buckets) == 10
    for b in cf.buckets:
        assert b.size == 3
    assert cf.size == 0


def test_repr_contains_expected_info():
    cf = cuckoofilter.CuckooFilter(capacity=5, fingerprint_size=1)
    r = repr(cf)
    assert "capacity=5" in r
    assert "size=0" in r
    assert "fingerprint size=1" in r
    # size should update after insert
    cf.insert("test")
    r2 = repr(cf)
    assert "size=1" in r2


def test_len_returns_size():
    cf = cuckoofilter.CuckooFilter(capacity=5)
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1
    cf.insert("b")
    assert len(cf) == 2


def test_contains_and_insert_consistency():
    cf = cuckoofilter.CuckooFilter(capacity=100)
    assert "missing" not in cf
    inserted = cf.insert("missing")
    assert inserted is True
    assert "missing" in cf
    # Using 'in' operator calls __contains__
    assert cf.__contains__("missing")
    # contains() method should return True and be consistent with 'in'
    assert cf.contains("missing") is True


def test_delete_after_insert():
    cf = cuckoofilter.CuckooFilter(capacity=50)
    item = "hello"
    assert cf.delete(item) is False  # deleting non-existing returns False
    cf.insert(item)
    assert cf.contains(item) is True
    result = cf.delete(item)
    assert result is True
    assert cf.contains(item) is False
    # size updated appropriately
    assert len(cf) == 0


def test_delete_only_affects_existing_items():
    cf = cuckoofilter.CuckooFilter(capacity=50)
    item1 = "test1"
    item2 = "test2"
    cf.insert(item1)
    assert cf.delete(item2) is False
    # item1 still present
    assert cf.contains(item1)
    # item2 never present
    assert not cf.contains(item2)


def test_insert_returns_true_or_raises_on_full(monkeypatch):
    # Patch max_displacements to low value to trigger full easily
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=1)

    # Insert two distinct items should succeed
    assert cf.insert("A") is True
    # Due to randomness, second insert might succeed too, but let's do one guaranteed occupy bucket
    assert cf.insert("B") is True or cf.insert("B") is True  # Just call once to occupy

    # Patch bucket insert to always fail to simulate full filter
    def fail_insert(self, _):
        return False

    def swap_return_same(self, fingerprint):
        return fingerprint

    # Patch insert and swap methods to simulate full bucket and no evictions
    for b in cf.buckets:
        monkeypatch.setattr(b, "insert", fail_insert.__get__(b))
        monkeypatch.setattr(b, "swap", swap_return_same.__get__(b))

    with pytest.raises(CuckooFilterFullException):
        cf.insert("C")


def test_alternate_index_consistency():
    cf = cuckoofilter.CuckooFilter(capacity=10, fingerprint_size=2)
    item = "myitem"
    fingerprint = cuckoofilter.hashutils.fingerprint(item, cf.fingerprint_size)
    i = cf._get_index(item)
    alt1 = cf._get_alternate_index(i, fingerprint)
    alt2 = cf._get_alternate_index(i, fingerprint)
    assert alt1 == alt2
    assert 0 <= alt1 < cf.capacity
    # Alternate index should differ from original index sometimes
    # But may be equal in edge cases
    # Just run multiple times on different inputs
    indexes = set()
    for s in ["a", "b", "c", "d"]:
        i = cf._get_index(s)
        f = cuckoofilter.hashutils.fingerprint(s, cf.fingerprint_size)
        alt = cf._get_alternate_index(i, f)
        assert 0 <= i < cf.capacity
        assert 0 <= alt < cf.capacity
        indexes.add(i)
        indexes.add(alt)
    assert len(indexes) >= 2  # We have some variety


def test_insert_and_delete_cross_bucket():
    # This test forces insert and delete using two buckets
    cf = cuckoofilter.CuckooFilter(capacity=5, bucket_size=2, max_displacements=10)
    # Insert multiple items
    items = ["alpha", "beta", "gamma", "delta"]
    for item in items:
        assert cf.insert(item) is True

    for item in items:
        assert cf.contains(item) is True

    # Delete some
    assert cf.delete("beta") is True
    assert cf.contains("beta") is False
    assert len(cf) == len(items) - 1

    # Delete again deleted item -> False
    assert cf.delete("beta") is False


def test_insert_handles_eviction_path(monkeypatch):
    # Setup a filter with a bucket size 1 and small capacity to force eviction
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=10)

    # Insert first item always succeeds
    assert cf.insert("first") is True

    # Patch random.choice to always pick first index to control eviction start
    called_indices = []
    def fixed_choice(seq):
        called_indices.append(seq)
        return seq[0]

    monkeypatch.setattr(random, "choice", fixed_choice)

    # Insert items until eviction occurs but still succeeds
    # We patch bucket.insert to accept on second try
    # Monkeypatch buckets insert for first bucket to always fail except the second bucket succeeds
    orig_insert_0 = cf.buckets[0].insert
    orig_insert_1 = cf.buckets[1].insert

    def insert_0_fail(fingerprint):
        # Always fail insert for bucket 0
        return False

    def insert_1_succeed(fingerprint):
        # accept only on insert call count > 0
        return True

    monkeypatch.setattr(cf.buckets[0], "insert", insert_0_fail)
    monkeypatch.setattr(cf.buckets[1], "insert", insert_1_succeed)

    # Also patch swap to just swap fingerprint without change
    def swap_keep(self, fingerprint):
        # return fingerprint to try next bucket insert
        return fingerprint

    monkeypatch.setattr(cf.buckets[0], "swap", swap_keep.__get__(cf.buckets[0]))
    monkeypatch.setattr(cf.buckets[1], "swap", swap_keep.__get__(cf.buckets[1]))

    # Inserting new item triggers eviction path but then insert succeeds
    result = cf.insert("second")
    assert result is True
    # size reflects two inserted items
    assert len(cf) == 2
    # random.choice indeed called with [i, j]
    assert any(isinstance(c, list) and len(c) == 2 for c in called_indices)


def test_insert_fingerprints_consistency():
    # Verify that fingerprint size matches requested
    for sz in [1, 2, 4]:
        cf = cuckoofilter.CuckooFilter(capacity=10, fingerprint_size=sz)
        fp = cuckoofilter.hashutils.fingerprint("item", sz)
        assert isinstance(fp, bytes) or isinstance(fp, (bytes, bytearray))
        # Its length matches fingerprint_size in bytes
        assert len(fp) == sz


def test_size_updating_after_insert_and_delete():
    cf = cuckoofilter.CuckooFilter(capacity=50)
    keys = ["k1", "k2", "k3"]
    for k in keys:
        assert cf.insert(k)
    assert cf.size == len(keys)
    # Delete one
    assert cf.delete(keys[1]) is True
    assert cf.size == len(keys) - 1
    # Delete again returns False, size unchanged
    assert cf.delete(keys[1]) is False
    assert cf.size == len(keys) - 1


def test_insert_duplicate_items():
    cf = cuckoofilter.CuckooFilter(capacity=20)
    item = "dupitem"
    # Insert once ok
    assert cf.insert(item) is True
    # Insert again: cuckoo filter allows duplicates? Insert returns True if space, but size increments?
    result = cf.insert(item)
    assert result in (True, CuckooFilterFullException) or result is True
    # However, size must be incremented only if insert succeeds
    if result is True:
        assert cf.size >= 2
    else:
        # If it threw, test handled by exception elsewhere
        pass
    # The item must be contained after one insert
    assert cf.contains(item)


def test_delete_nonexistent_returns_false_and_size_unchanged():
    cf = cuckoofilter.CuckooFilter(capacity=10)
    initial_size = cf.size
    assert cf.delete("nonexistent") is False
    assert cf.size == initial_size


def test_insert_raises_for_full_filter_random_deterministic(monkeypatch):
    # Setup a filter with max_displacements=0 to force raising quickly
    cf = cuckoofilter.CuckooFilter(capacity=2, bucket_size=1, max_displacements=0)

    # Insert one to fill filter partially
    assert cf.insert("X") is True

    # Patch buckets insert to always return False (always full)
    def fail_insert(self, _):
        return False

    def swap_fingerprint(self, fingerprint):
        return fingerprint

    for b in cf.buckets:
        monkeypatch.setattr(b, "insert", fail_insert.__get__(b))
        monkeypatch.setattr(b, "swap", swap_fingerprint.__get__(b))

    # Because max_displacements=0, insert must raise immediately
    with pytest.raises(CuckooFilterFullException):
        cf.insert("Y")
