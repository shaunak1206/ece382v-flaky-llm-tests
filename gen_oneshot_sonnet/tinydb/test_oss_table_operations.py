"""Tests for the table API (insert, search, update, remove, ...) using MemoryStorage."""
import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.table import Document


@pytest.fixture
def db():
    database = TinyDB(storage=MemoryStorage)
    yield database
    database.close()


@pytest.fixture
def people(db):
    db.insert_multiple(
        [
            {"name": "John", "age": 30, "city": "NY"},
            {"name": "Jane", "age": 25, "city": "LA"},
            {"name": "Bob", "age": 40, "city": "NY"},
        ]
    )
    return db


def test_insert_returns_sequential_ids(db):
    assert db.insert({"a": 1}) == 1
    assert db.insert({"a": 2}) == 2
    assert len(db) == 2


@pytest.mark.parametrize("bad", [[1, 2], "text", 42, None])
def test_insert_rejects_non_mappings(db, bad):
    with pytest.raises(ValueError):
        db.insert(bad)
    assert len(db) == 0


def test_insert_document_uses_its_own_id(db):
    assert db.insert(Document({"a": 1}, doc_id=10)) == 10
    assert db.get(doc_id=10) == {"a": 1}
    assert db.insert({"a": 2}) == 11


def test_insert_duplicate_document_id_is_rejected(db):
    db.insert(Document({"a": 1}, doc_id=5))
    with pytest.raises(ValueError):
        db.insert(Document({"a": 2}, doc_id=5))
    assert db.get(doc_id=5) == {"a": 1}
    assert len(db) == 1


def test_insert_multiple_returns_ids_in_order(db):
    assert db.insert_multiple([{"a": i} for i in range(4)]) == [1, 2, 3, 4]
    assert db.insert_multiple([]) == []
    assert len(db) == 4


def test_insert_multiple_accepts_documents_with_ids(db):
    ids = db.insert_multiple([Document({"a": 1}, doc_id=20), {"a": 2}])
    assert ids[0] == 20
    assert ids[1] != 20
    assert len(db) == 2


def test_insert_multiple_with_invalid_item_inserts_nothing(db):
    with pytest.raises(ValueError):
        db.insert_multiple([{"a": 1}, "oops"])
    assert len(db) == 0


def test_all_returns_documents_with_ids(people):
    docs = people.all()
    assert [d["name"] for d in docs] == ["John", "Jane", "Bob"]
    assert [d.doc_id for d in docs] == [1, 2, 3]
    assert all(isinstance(d, Document) for d in docs)


def test_get_by_condition_and_by_id(people):
    doc = people.get(where("name") == "Jane")
    assert doc == {"name": "Jane", "age": 25, "city": "LA"}
    assert doc.doc_id == 2
    assert people.get(where("name") == "Nobody") is None
    assert people.get(doc_id=3)["name"] == "Bob"
    assert people.get(doc_id=3).doc_id == 3
    assert people.get(doc_id=99) is None


def test_get_with_multiple_ids_returns_a_list(people):
    docs = people.get(doc_ids=[1, 3])
    assert sorted(d.doc_id for d in docs) == [1, 3]
    assert people.get(doc_ids=[42]) == []


@pytest.mark.parametrize("method", ["get", "contains", "remove"])
def test_calls_without_arguments_raise(db, method):
    with pytest.raises(RuntimeError):
        getattr(db, method)()


def test_search_and_count(people):
    res = people.search(where("city") == "NY")
    assert [d["name"] for d in res] == ["John", "Bob"]
    assert people.search(where("age") > 100) == []
    assert people.count(where("age") >= 30) == 2
    assert people.count(where("age") > 100) == 0


def test_search_cache_is_invalidated_by_writes(people):
    query = where("city") == "NY"
    assert len(people.search(query)) == 2
    assert len(people.search(query)) == 2
    people.insert({"name": "Zed", "age": 1, "city": "NY"})
    assert len(people.search(query)) == 3
    people.remove(where("name") == "John")
    assert len(people.search(query)) == 2
    people.update({"city": "LA"}, where("name") == "Bob")
    assert len(people.search(query)) == 1


def test_search_returns_a_fresh_list_each_time(people):
    query = where("city") == "NY"
    first = people.search(query)
    first.clear()
    assert len(people.search(query)) == 2


def test_contains(people):
    assert people.contains(where("name") == "Bob")
    assert not people.contains(where("name") == "Eve")
    assert people.contains(doc_id=1)
    assert not people.contains(doc_id=50)


def test_update_with_dict(people):
    ids = people.update({"city": "SF"}, where("city") == "NY")
    assert ids == [1, 3]
    assert people.get(doc_id=1)["city"] == "SF"
    assert people.get(doc_id=1)["name"] == "John"
    assert people.get(doc_id=2)["city"] == "LA"
    ids = people.update({"flag": True})
    assert sorted(ids) == [1, 2, 3]
    assert all(d["flag"] for d in people.all())


def test_update_with_callable(people):
    def birthday(doc):
        doc["age"] += 1

    assert people.update(birthday, where("name") == "John") == [1]
    assert people.get(where("name") == "John")["age"] == 31
    assert people.get(where("name") == "Jane")["age"] == 25


def test_update_by_doc_ids_skips_missing_ids(people):
    assert people.update({"x": 1}, doc_ids=[1, 99]) == [1]
    assert people.get(doc_id=1)["x"] == 1
    assert people.get(doc_id=2).get("x") is None
    assert people.update({"x": 2}, doc_ids=[500]) == []


def test_update_multiple(people):
    people.update_multiple(
        [
            ({"tag": "young"}, where("age") < 30),
            ({"tag": "old"}, where("age") >= 40),
        ]
    )
    assert people.get(doc_id=2)["tag"] == "young"
    assert people.get(doc_id=3)["tag"] == "old"
    assert "tag" not in people.get(doc_id=1)


def test_upsert_updates_or_inserts(people):
    assert people.upsert({"age": 31}, where("name") == "John") == [1]
    assert len(people) == 3
    assert people.get(doc_id=1)["age"] == 31
    assert people.upsert({"name": "New", "age": 1}, where("name") == "New") == [4]
    assert len(people) == 4
    assert people.get(doc_id=4)["name"] == "New"


def test_upsert_with_document_id(db):
    assert db.upsert(Document({"a": 1}, doc_id=7)) == [7]
    assert db.upsert(Document({"a": 2}, doc_id=7)) == [7]
    assert db.get(doc_id=7) == {"a": 2}
    assert len(db) == 1


def test_upsert_without_condition_or_id_raises(db):
    with pytest.raises(ValueError):
        db.upsert({"a": 1})


def test_remove_by_condition_and_by_ids(people):
    assert people.remove(where("city") == "NY") == [1, 3]
    assert [d["name"] for d in people.all()] == ["Jane"]
    assert people.remove(doc_ids=[2, 77]) == [2]
    assert len(people) == 0


def test_truncate_resets_ids_and_removal_never_reuses_live_ids(people):
    people.remove(doc_ids=[1])
    new_id = people.insert({"name": "X"})
    assert new_id not in (2, 3)
    assert len(people) == 3
    people.truncate()
    assert len(people) == 0
    assert people.insert({"a": 1}) == 1


def test_table_basics(db):
    table = db.table("things")
    assert table.name == "things"
    assert table.storage is db.storage
    table.insert_multiple([{"i": 1}, {"i": 2}])
    assert len(table) == 2
    assert [d["i"] for d in table] == [1, 2]
    assert "things" in repr(table)
    assert "total=2" in repr(table)
    assert len(db.table("other")) == 0
    d = Document({"a": 1}, doc_id=3)
    assert d.doc_id == 3
    assert d == {"a": 1}


def test_query_cache_is_bounded(db):
    table = db.table("t", cache_size=2)
    table.insert_multiple([{"v": i} for i in range(5)])
    for i in range(4):
        table.search(where("v") == i)
    assert len(table._query_cache) == 2
    table.clear_cache()
    assert len(table._query_cache) == 0
