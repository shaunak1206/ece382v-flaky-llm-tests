"""
Unit tests for tinydb.database.TinyDB and tinydb.table.Table, covering
JSONStorage-backed (real temp files) and MemoryStorage-backed usage:
insert, insert_multiple, all, get, search, update, upsert, remove,
truncate, contains, count, and doc_id handling.
"""
import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document


@pytest.fixture
def json_path(tmp_path):
    return str(tmp_path / "db.json")


@pytest.fixture(params=["json", "memory"])
def db(request, tmp_path):
    if request.param == "json":
        path = str(tmp_path / "data.json")
        instance = TinyDB(path, storage=JSONStorage)
    else:
        instance = TinyDB(storage=MemoryStorage)
    yield instance
    instance.close()


def test_insert_returns_doc_id(db):
    doc_id = db.insert({"name": "John", "age": 22})
    assert isinstance(doc_id, int)
    assert doc_id == 1


def test_insert_multiple_returns_ids(db):
    ids = db.insert_multiple([{"x": 1}, {"x": 2}, {"x": 3}])
    assert ids == [1, 2, 3]
    assert len(db) == 3


def test_insert_non_mapping_raises(db):
    with pytest.raises(ValueError):
        db.insert("not-a-mapping")


def test_all_returns_all_documents(db):
    db.insert({"a": 1})
    db.insert({"a": 2})
    docs = db.all()
    assert len(docs) == 2
    assert {d["a"] for d in docs} == {1, 2}


def test_get_by_doc_id(db):
    doc_id = db.insert({"name": "Alice"})
    doc = db.get(doc_id=doc_id)
    assert doc["name"] == "Alice"
    assert doc.doc_id == doc_id


def test_get_by_doc_id_missing_returns_none(db):
    assert db.get(doc_id=999) is None


def test_get_by_query(db):
    db.insert({"name": "Alice"})
    db.insert({"name": "Bob"})
    doc = db.get(where("name") == "Bob")
    assert doc["name"] == "Bob"


def test_get_by_doc_ids(db):
    id1 = db.insert({"n": 1})
    id2 = db.insert({"n": 2})
    db.insert({"n": 3})
    docs = db.get(doc_ids=[id1, id2])
    assert sorted(d["n"] for d in docs) == [1, 2]


def test_search_returns_matching_documents(db):
    db.insert({"name": "Alice", "age": 30})
    db.insert({"name": "Bob", "age": 20})
    result = db.search(where("age") >= 25)
    assert len(result) == 1
    assert result[0]["name"] == "Alice"


def test_search_uses_cache_and_is_independent_copy(db):
    db.insert({"a": 1})
    q = where("a") == 1
    result1 = db.search(q)
    result2 = db.search(q)
    assert result1 == result2
    assert result1 is not result2  # fresh list copy each time


def test_contains_by_query(db):
    db.insert({"name": "Alice"})
    assert db.contains(where("name") == "Alice") is True
    assert db.contains(where("name") == "Nobody") is False


def test_contains_by_doc_id(db):
    doc_id = db.insert({"name": "Alice"})
    assert db.contains(doc_id=doc_id) is True
    assert db.contains(doc_id=doc_id + 100) is False


def test_count(db):
    db.insert({"type": "a"})
    db.insert({"type": "a"})
    db.insert({"type": "b"})
    assert db.count(where("type") == "a") == 2
    assert db.count(where("type") == "c") == 0


def test_update_with_fields_by_query(db):
    db.insert({"name": "Alice", "age": 30})
    updated_ids = db.update({"age": 31}, where("name") == "Alice")
    assert len(updated_ids) == 1
    assert db.get(doc_id=updated_ids[0])["age"] == 31


def test_update_with_callable(db):
    doc_id = db.insert({"count": 1})

    def increment(doc):
        doc["count"] += 10

    db.update(increment, where("count") == 1)
    assert db.get(doc_id=doc_id)["count"] == 11


def test_update_by_doc_ids(db):
    id1 = db.insert({"a": 1})
    id2 = db.insert({"a": 2})
    updated = db.update({"a": 100}, doc_ids=[id1, id2])
    assert sorted(updated) == sorted([id1, id2])
    assert db.get(doc_id=id1)["a"] == 100
    assert db.get(doc_id=id2)["a"] == 100


def test_update_by_doc_ids_skips_missing(db):
    id1 = db.insert({"a": 1})
    updated = db.update({"a": 5}, doc_ids=[id1, 99999])
    assert updated == [id1]


def test_upsert_inserts_when_no_match(db):
    ids = db.upsert({"name": "New", "age": 1}, where("name") == "New")
    assert len(ids) == 1
    assert db.get(doc_id=ids[0])["age"] == 1


def test_upsert_updates_when_match(db):
    doc_id = db.insert({"name": "Alice", "age": 30})
    ids = db.upsert({"age": 31}, where("name") == "Alice")
    assert ids == [doc_id]
    assert db.get(doc_id=doc_id)["age"] == 31
    assert len(db) == 1


def test_upsert_without_cond_or_doc_id_raises(db):
    with pytest.raises(ValueError):
        db.upsert({"name": "Nope"})


def test_remove_by_query(db):
    id1 = db.insert({"keep": True})
    id2 = db.insert({"keep": False})
    removed = db.remove(where("keep") == False)  # noqa: E712
    assert removed == [id2]
    assert db.contains(doc_id=id1)
    assert not db.contains(doc_id=id2)


def test_remove_by_doc_ids(db):
    id1 = db.insert({"a": 1})
    id2 = db.insert({"a": 2})
    removed = db.remove(doc_ids=[id1])
    assert removed == [id1]
    assert len(db) == 1
    assert db.contains(doc_id=id2)


def test_truncate_removes_all_documents(db):
    db.insert({"a": 1})
    db.insert({"a": 2})
    db.truncate()
    assert len(db) == 0
    assert db.all() == []


def test_truncate_resets_doc_id_counter(db):
    db.insert({"a": 1})
    db.insert({"a": 2})
    db.truncate()
    new_id = db.insert({"a": 3})
    assert new_id == 1


def test_len_and_iter(db):
    db.insert({"a": 1})
    db.insert({"a": 2})
    assert len(db) == 2
    assert sorted(doc["a"] for doc in db) == [1, 2]


def test_doc_ids_increase_and_are_not_reused_after_removal(db):
    id1 = db.insert({"a": 1})
    id2 = db.insert({"a": 2})
    db.remove(doc_ids=[id2])
    id3 = db.insert({"a": 3})
    assert id3 == id2 + 1
    assert id1 == 1


def test_insert_document_with_explicit_doc_id(db):
    doc = Document({"a": 1}, doc_id=42)
    inserted_id = db.insert(doc)
    assert inserted_id == 42
    assert db.get(doc_id=42)["a"] == 1


def test_multiple_tables_are_independent(db):
    db.insert({"a": 1})
    other = db.table("other")
    other.insert({"b": 2})

    assert len(db) == 1
    assert len(other) == 1
    assert db.all()[0]["a"] == 1
    assert other.all()[0]["b"] == 2


def test_drop_table(db):
    db.table("temp").insert({"a": 1})
    assert "temp" in db.tables()
    db.drop_table("temp")
    assert "temp" not in db.tables()


# --- JSONStorage-specific: real temp file persistence -----------------------

def test_json_storage_persists_across_instances(json_path):
    with TinyDB(json_path, storage=JSONStorage) as db1:
        db1.insert({"name": "Persisted"})

    with TinyDB(json_path, storage=JSONStorage) as db2:
        docs = db2.all()
        assert len(docs) == 1
        assert docs[0]["name"] == "Persisted"


