import pytest

from cuckoopy.cuckoofilter import CuckooFilter
from cuckoopy.exceptions import CuckooFilterFullException


@pytest.fixture
def hash_setup(monkeypatch):
    """
    Patch hashutils so tests can use deterministic item -> fingerprint/index
    mappings without relying on the project's hashing implementation.
    """
    item_to_fp = {}
    fp_to_code = {}
    item_to_code = {}

    def fingerprint(item, size):
        return item_to_fp[item]

    def hash_code(obj):
        if obj in item_to_code:
            return item_to_code[obj]
        if obj in fp_to_code:
            return fp_to_code[obj]
        return 0

    monkeypatch.setattr("cuckoopy.cuckoofilter.hashutils.fingerprint", fingerprint)
    monkeypatch.setattr("cuckoopy.cuckoofilter.hashutils.hash_code", hash_code)

    return item_to_fp, item_to_code, fp_to_code


def test_exception_is_subclass_of_exception():
    assert issubclass(CuckooFilterFullException, Exception)


def test_init_sets_capacity_bucket_size_and_creates_buckets():
    cf = CuckooFilter(capacity=5, bucket_size=3, fingerprint_size=2, max_displacements=7)
    assert cf.capacity == 5
    assert cf.bucket_size == 3
    assert cf.fingerprint_size == 2
    assert cf.max_displacements == 7
    assert len(cf.buckets) == 5
    assert all(len(bucket.slots) == 3 for bucket in cf.buckets)
    assert len(cf) == 0


def test_repr_includes_key_fields():
    cf = CuckooFilter(capacity=4, bucket_size=2, fingerprint_size=8)
    text = repr(cf)
    assert "<CuckooFilter:" in text
    assert "capacity=4" in text
    assert "size=0" in text
    assert "fingerprint size=8 byte(s)" in text


def test_insert_returns_true_and_increments_size_when_primary_bucket_succeeds(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_code["a"] = 0
    fp_to_code["fp-a"] = 1

    cf = CuckooFilter(capacity=4, bucket_size=2)
    assert cf.insert("a") is True
    assert len(cf) == 1
    assert cf.contains("a") is True


def test_contains_uses_in_operator_alias(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_code["a"] = 0
    fp_to_code["fp-a"] = 1

    cf = CuckooFilter(capacity=4, bucket_size=2)
    cf.insert("a")
    assert "a" in cf
    assert cf.contains("a") is True


def test_contains_returns_false_for_missing_item(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["missing"] = "fp-missing"
    item_to_code["a"] = 0
    item_to_code["missing"] = 1
    fp_to_code["fp-a"] = 1
    fp_to_code["fp-missing"] = 2

    cf = CuckooFilter(capacity=4, bucket_size=2)
    cf.insert("a")
    assert cf.contains("missing") is False
    assert "missing" not in cf


def test_insert_uses_alternate_bucket_when_primary_is_full(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["b"] = "fp-b"
    item_to_code["a"] = 0
    item_to_code["b"] = 0
    fp_to_code["fp-a"] = 1
    fp_to_code["fp-b"] = 2

    cf = CuckooFilter(capacity=4, bucket_size=1)
    assert cf.insert("a") is True
    assert cf.insert("b") is True
    assert len(cf) == 2
    assert cf.contains("a") is True
    assert cf.contains("b") is True


def test_delete_returns_true_and_decrements_size_when_item_present(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_code["a"] = 0
    fp_to_code["fp-a"] = 1

    cf = CuckooFilter(capacity=4, bucket_size=2)
    cf.insert("a")
    assert cf.delete("a") is True
    assert len(cf) == 0
    assert cf.contains("a") is False


def test_delete_returns_false_for_missing_item(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["missing"] = "fp-missing"
    item_to_code["a"] = 0
    item_to_code["missing"] = 1
    fp_to_code["fp-a"] = 1
    fp_to_code["fp-missing"] = 2

    cf = CuckooFilter(capacity=4, bucket_size=2)
    cf.insert("a")
    assert cf.delete("missing") is False
    assert len(cf) == 1


def test_delete_finds_item_in_alternate_bucket(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["b"] = "fp-b"
    item_to_code["a"] = 0
    item_to_code["b"] = 0
    fp_to_code["fp-a"] = 1
    fp_to_code["fp-b"] = 2

    cf = CuckooFilter(capacity=4, bucket_size=1)
    cf.insert("a")
    cf.insert("b")
    assert cf.delete("b") is True
    assert cf.contains("b") is False
    assert len(cf) == 1


def test_len_tracks_number_of_successful_operations(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["b"] = "fp-b"
    item_to_code["a"] = 0
    item_to_code["b"] = 0
    fp_to_code["fp-a"] = 1
    fp_to_code["fp-b"] = 2

    cf = CuckooFilter(capacity=4, bucket_size=1)
    assert len(cf) == 0
    cf.insert("a")
    assert len(cf) == 1
    cf.insert("b")
    assert len(cf) == 2
    cf.delete("a")
    assert len(cf) == 1


def test_bucket_size_property_is_preserved_on_filter_instance():
    cf = CuckooFilter(capacity=3, bucket_size=7)
    assert cf.bucket_size == 7
    assert all(len(bucket.slots) == 7 for bucket in cf.buckets)


def test_insert_raises_full_exception_when_no_space_and_no_displacement_possible(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["b"] = "fp-b"
    item_to_code["a"] = 0
    item_to_code["b"] = 0
    fp_to_code["fp-a"] = 0
    fp_to_code["fp-b"] = 0

    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=1)
    assert cf.insert("a") is True

    with pytest.raises(CuckooFilterFullException) as excinfo:
        cf.insert("b")

    assert "Filter is full" in str(excinfo.value)
    assert len(cf) == 1


def test_insert_does_not_change_size_when_it_raises_full_exception(hash_setup):
    item_to_fp, item_to_code, fp_to_code = hash_setup
    item_to_fp["a"] = "fp-a"
    item_to_fp["b"] = "fp-b"
    item_to_code["a"] = 0
    item_to_code["b"] = 0
    fp_to_code["fp-a"] = 0
    fp_to_code["fp-b"] = 0

    cf = CuckooFilter(capacity=1, bucket_size=1, max_displacements=0)
    cf.insert("a")
    with pytest.raises(CuckooFilterFullException):
        cf.insert("b")
    assert len(cf) == 1
    assert cf.contains("a") is True
