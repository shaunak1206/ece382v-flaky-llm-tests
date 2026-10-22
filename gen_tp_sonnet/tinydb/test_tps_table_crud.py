import pytest
from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.table import Document


@pytest.fixture
def db():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([
        {'name': 'John', 'age': 22},
        {'name': 'John', 'age': 37},
        {'name': 'Megan', 'age': 27},
        {'name': 'Bob', 'age': 42},
    ])
    return database


def test_insert_multiple_returns_ids(db):
    ids = db.insert_multiple([{'x': 1}, {'x': 2}])
    assert ids == [5, 6]


def test_insert_rejects_non_mapping():
    database = TinyDB(storage=MemoryStorage)
    with pytest.raises(ValueError):
        database.insert([1, 2, 3])


def test_all_returns_documents_with_doc_ids(db):
    docs = db.all()
    assert [d.doc_id for d in docs] == [1, 2, 3, 4]
    assert all(isinstance(d, Document) for d in docs)


def test_search_returns_matching_documents(db):
    result = db.search(where('name') == 'John')
    assert result == [{'name': 'John', 'age': 22}, {'name': 'John', 'age': 37}]


def test_search_reflects_changes_after_write(db):
    query = where('name') == 'Alice'
    assert db.search(query) == []
    db.insert({'name': 'Alice', 'age': 30})
    assert db.search(query) == [{'name': 'Alice', 'age': 30}]


def test_get_by_cond_returns_a_single_document(db):
    doc = db.get(where('name') == 'Megan')
    assert doc == {'name': 'Megan', 'age': 27}
    assert doc.doc_id == 3


def test_get_by_doc_id(db):
    assert db.get(doc_id=4) == {'name': 'Bob', 'age': 42}


def test_get_missing_returns_none(db):
    assert db.get(where('name') == 'Nobody') is None
    assert db.get(doc_id=999) is None


def test_get_with_multiple_doc_ids_returns_list(db):
    docs = db.get(doc_ids=[1, 4])
    assert isinstance(docs, list)
    assert [d['name'] for d in docs] == ['John', 'Bob']


def test_get_without_arguments_raises(db):
    with pytest.raises(RuntimeError):
        db.get()


def test_contains_by_cond(db):
    assert db.contains(where('name') == 'Bob')
    assert not db.contains(where('name') == 'Zed')


def test_contains_by_doc_id(db):
    assert db.contains(doc_id=1)
    assert not db.contains(doc_id=100)


def test_count_matches(db):
    assert db.count(where('name') == 'John') == 2
    assert db.count(where('name') == 'Nobody') == 0
    assert db.count(where('age') >= 0) == 4


def test_update_with_cond_changes_matching_documents(db):
    ids = db.update({'age': 99}, where('name') == 'John')
    assert sorted(ids) == [1, 2]
    assert db.count(where('age') == 99) == 2
    assert db.get(doc_id=3)['age'] == 27


def test_update_without_cond_updates_all_documents(db):
    ids = db.update({'flag': True})
    assert sorted(ids) == [1, 2, 3, 4]
    assert all(d['flag'] is True for d in db.all())


def test_update_with_doc_ids_skips_unknown_ids(db):
    ids = db.update({'age': 1}, doc_ids=[1, 999])
    assert ids == [1]
    assert db.get(doc_id=1)['age'] == 1


def test_update_with_callable_mutates_document(db):
    def birthday(doc):
        doc['age'] += 1

    db.update(birthday, where('name') == 'Bob')
    assert db.get(doc_id=4)['age'] == 43


def test_update_multiple_applies_each_update(db):
    ids = db.update_multiple([
        ({'age': 1}, where('name') == 'Megan'),
        ({'age': 2}, where('name') == 'Bob'),
    ])
    assert sorted(ids) == [3, 4]
    assert db.get(doc_id=3)['age'] == 1
    assert db.get(doc_id=4)['age'] == 2


def test_upsert_updates_existing_documents(db):
    ids = db.upsert({'age': 50}, where('name') == 'Bob')
    assert ids == [4]
    assert db.count(where('name') == 'Bob') == 1
    assert db.get(doc_id=4)['age'] == 50


def test_upsert_inserts_when_nothing_matches(db):
    ids = db.upsert({'name': 'Zed', 'age': 5}, where('name') == 'Zed')
    assert ids == [5]
    assert db.get(doc_id=5) == {'name': 'Zed', 'age': 5}


def test_remove_by_cond_returns_removed_ids(db):
    ids = db.remove(where('name') == 'John')
    assert sorted(ids) == [1, 2]
    assert db.count(where('name') == 'John') == 0
    assert len(db.all()) == 2


def test_remove_by_doc_ids_skips_unknown(db):
    ids = db.remove(doc_ids=[2, 999])
    assert ids == [2]
    assert db.get(doc_id=2) is None
    assert len(db.all()) == 3


def test_remove_without_arguments_raises(db):
    with pytest.raises(RuntimeError):
        db.remove()


def test_truncate_removes_all_documents(db):
    db.truncate()
    assert db.all() == []
    assert db.count(where('age') > 0) == 0
    db.insert({'new': 1})
    assert len(db.all()) == 1
