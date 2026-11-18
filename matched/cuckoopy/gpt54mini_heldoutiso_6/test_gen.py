import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


class FakeBucket:
    def __init__(self, size=4, initial=None, insert_result=True, delete_result=True, swap_result=None):
        self.size = size
        self.items = list(initial or [])
        self.insert_result = insert_result
        self.delete_result = delete_result
        self.swap_result = swap_result

    def insert(self, item):
        if not self.insert_result:
            return False
        if len(self.items) >= self.size:
            return False
        self.items.append(item)
        return True

    def delete(self, item):
        if item in self.items:
            self.items.remove(item)
            return True
        return False

    def swap(self, item):
        if self.swap_result is not None:
            return self.swap_result
        if not self.items:
            self.items.append(item)
            return item
        replaced = self.items[0]
        self.items[0] = item
        return replaced

    def __contains__(self, item):
        return item in self.items


@pytest.fixture
def deterministic_hashes(monkeypatch):
    from cuckoopy import cuckoofilter as cf_mod

    def fake_hash_code(value):
        mapping = {
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
            "epsilon": 5,
            "zeta": 6,
            "theta": 7,
            "iota": 8,
        }
        return mapping.get(value, value if isinstance(value, int) else 0)

    def fake_fingerprint(item, size):
        mapping = {
            "alpha": b"a",
            "beta": b"b",
            "gamma": b"g",
            "delta": b"d",
            "epsilon": b"e",
            "zeta": b"z",
            "theta": b"t",
            "iota": b"i",
        }
        return mapping.get(item, bytes([item % 256]) if isinstance(item, int) else b"x")[:size]

    monkeypatch.setattr(cf_mod.hashutils, "hash_code", fake_hash_code)
    monkeypatch.setattr(cf_mod.hashutils, "fingerprint", fake_fingerprint)


def test_init_sets_capacity_bucket_size_and_empty_size():
    f = CuckooFilter(capacity=7, bucket_size=3, fingerprint_size=2, max_displacements=11)

    assert f.capacity == 7
    assert f.bucket_size == 3
    assert f.fingerprint_size == 2
    assert f.max_displacements == 11
    assert len(f.buckets) == 7
    assert len(f) == 0


def test_repr_includes_capacity_size_and_fingerprint_size():
    f = CuckooFilter(capacity=5, bucket_size=2, fingerprint_size=4)

    assert repr(f) == "<CuckooFilter: capacity=5, size=0, fingerprint size=4 byte(s)>"


def test_bucket_size_attribute_reflects_constructor_argument():
    f = CuckooFilter(capacity=3, bucket_size=9)

    assert f.bucket_size == 9
    assert all(bucket.size == 9 for bucket in f.buckets)


def test_contains_uses_overloaded_in_operator(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]
    f.buckets[1].items.append(b"a")

    assert ("alpha" in f) is True
    assert ("beta" in f) is False


def test_insert_places_item_in_primary_bucket_when_available(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]

    assert f.insert("alpha") is True
    assert len(f) == 1
    assert b"a" in f.buckets[1]
    assert f.contains("alpha") is True


def test_insert_places_item_in_alternate_bucket_when_primary_is_full(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)
    f.buckets = [FakeBucket(size=1) for _ in range(4)]
    f.buckets[1].items.append(b"a")  # primary full
    assert not f.buckets[3].items     # alternate empty for "alpha" with our hashes

    assert f.insert("alpha") is True
    assert len(f) == 1
    assert b"a" in f.buckets[3]
    assert f.contains("alpha") is True


def test_insert_same_item_twice_increments_size_twice(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]

    assert f.insert("alpha") is True
    assert f.insert("alpha") is True

    assert len(f) == 2
    assert f.contains("alpha") is True


def test_delete_removes_item_from_primary_bucket_and_decrements_size(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]
    f.insert("alpha")
    assert len(f) == 1

    assert f.delete("alpha") is True
    assert len(f) == 0
    assert f.contains("alpha") is False


def test_delete_removes_item_from_alternate_bucket_when_not_in_primary(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=1, fingerprint_size=1)
    f.buckets = [FakeBucket(size=1) for _ in range(4)]
    f.buckets[3].items.append(b"a")

    assert f.delete("alpha") is True
    assert len(f) == 0
    assert b"a" not in f.buckets[3]


def test_delete_returns_false_for_missing_item_and_does_not_change_size(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]
    f.insert("alpha")
    before = len(f)

    assert f.delete("beta") is False
    assert len(f) == before
    assert f.contains("alpha") is True


def test_contains_returns_false_for_missing_item(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=1)
    f.buckets = [FakeBucket(size=2) for _ in range(4)]
    f.insert("alpha")

    assert f.contains("beta") is False


def test_get_alternate_index_matches_xor_hash_formula(monkeypatch, deterministic_hashes):
    f = CuckooFilter(capacity=8, bucket_size=2, fingerprint_size=1)
    fingerprint = b"a"
    index = 1

    expected = (index ^ 1) % 8
    assert f._get_alternate_index(index, fingerprint) == expected


def test_insert_raises_full_exception_when_no_space_after_displacements(monkeypatch, deterministic_hashes):
    from cuckoopy import cuckoofilter as cf_mod

    f = CuckooFilter(capacity=2, bucket_size=1, fingerprint_size=1, max_displacements=3)
    f.buckets = [
        FakeBucket(size=1, initial=[b"x"], insert_result=False, swap_result=b"y"),
        FakeBucket(size=1, initial=[b"z"], insert_result=False, swap_result=b"y"),
    ]

    monkeypatch.setattr(cf_mod.random, "choice", lambda seq: seq[0])

    with pytest.raises(CuckooFilterFullException) as excinfo:
        f.insert("alpha")

    assert "Filter is full" in str(excinfo.value)
    assert len(f) == 0


def test_cuckoo_filter_full_exception_is_subclass_of_exception():
    assert issubclass(CuckooFilterFullException, Exception)


def test_exception_can_be_instantiated_and_stringified():
    exc = CuckooFilterFullException("Insert operation failed. Filter is full.")
    assert isinstance(exc, Exception)
    assert "Filter is full" in str(exc)
