import pytest

from tinydb import Query, TinyDB, where
from tinydb.operations import increment
from tinydb.storages import MemoryStorage
from tinydb.table import Document, Table


@pytest.fixture
def db():
    return TinyDB(storage=MemoryStorage)


@pytest.fixture
def table(db):
    t = db.table('people')
    t.insert_multiple([
        {'name': 'alice', 'age': 30},
        {'name': 'bob', 'age': 25},
        {'name': 'carol', 'age': 35},
    ])
    return t


def test_insert_returns_incrementing_ids_and_stores_a_copy(db):
    t = db.table('t')
    original = {'a': 1}
    assert t.insert(original) == 1
    assert t.insert({'a': 2}) == 2
    assert t.insert({'a': 3}) == 3
    original['a'] = 99
    assert t.get(doc_id=1) == {'a': 1}


def test_insert_rejects_non_mapping(db):
    t = db.table('t')
    with pytest.raises(ValueError):
        t.insert([1, 2, 3])
    with pytest.raises(ValueError):
        t.insert('a string')
    assert len(t) == 0


def test_insert_multiple_returns_ids_and_is_all_or_nothing(db):
    bad = db.table('bad')
    with pytest.raises(ValueError):
        bad.insert_multiple([{'a': 1}, 5])
    assert len(bad) == 0
    t = db.table('t')
    assert t.insert_multiple([{'a': 1}, {'a': 2}, {'a': 3}]) == [1, 2, 3]
    assert [d['a'] for d in t.all()] == [1, 2, 3]


def test_insert_document_uses_its_own_doc_id(db):
    t = db.table('t')
    assert t.insert(Document({'a': 1}, 10)) == 10
    assert t.get(doc_id=10) == {'a': 1}
    # next plain insert must not collide with the explicit id
    assert t.insert({'a': 2}) == 11


def test_insert_duplicate_document_id_raises(db):
    t = db.table('t')
    t.insert({'a': 1})
    with pytest.raises(ValueError, match='already exists'):
        t.insert(Document({'a': 2}, 1))
    with pytest.raises(ValueError, match='already exists'):
        t.insert_multiple([Document({'a': 3}, 1)])
    assert len(t) == 1


def test_ids_continue_from_storage_and_are_not_reused_after_remove(db):
    t1 = Table(db.storage, 'shared')
    t1.insert_multiple([{'a': 1}, {'a': 2}])
    t2 = Table(db.storage, 'shared')
    assert t2.insert({'a': 3}) == 3
    t2.remove(doc_ids=[3])
    assert t2.insert({'a': 4}) == 4


def test_get_by_doc_id_and_missing_id(table):
    doc = table.get(doc_id=2)
    assert doc == {'name': 'bob', 'age': 25}
    assert isinstance(doc, Document)
    assert doc.doc_id == 2
    assert table.get(doc_id=999) is None


def test_get_by_cond_returns_first_match_or_none(table):
    doc = table.get(where('age') > 26)
    assert doc is not None
    assert doc['name'] == 'alice'
    assert doc.doc_id == 1
    assert table.get(where('name') == 'nobody') is None
    assert table.get(cond=where('name') == 'bob').doc_id == 2


def test_get_by_multiple_doc_ids_and_no_arguments(table):
    docs = table.get(doc_ids=[1, 3])
    assert [d.doc_id for d in docs] == [1, 3]
    assert [d['name'] for d in docs] == ['alice', 'carol']
    assert table.get(doc_ids=[]) == []
    assert table.get(doc_ids=[42]) == []
    with pytest.raises(RuntimeError):
        table.get()


def test_contains_by_cond_and_by_id(table):
    assert table.contains(where('name') == 'alice')
    assert not table.contains(where('name') == 'zed')
    assert table.contains(doc_id=1)
    assert not table.contains(doc_id=77)
    with pytest.raises(RuntimeError):
        table.contains()


def test_all_len_and_iter_agree(table):
    assert len(table) == 3
    assert len(table.all()) == 3
    assert [d.doc_id for d in table] == [1, 2, 3]
    assert all(isinstance(d, Document) for d in table)


def test_search_and_count(table):
    results = table.search(where('age') >= 30)
    assert [d['name'] for d in results] == ['alice', 'carol']
    assert [d.doc_id for d in results] == [1, 3]
    assert table.search(where('age') > 100) == []
    assert table.count(where('age') > 26) == 2
    assert table.count(where('age') > 100) == 0


def test_search_is_cached_and_cache_cleared_on_write(table):
    q = where('age') > 26
    first = table.search(q)
    assert table._query_cache.length == 1
    assert table.search(q) == first
    table.insert({'name': 'dave', 'age': 50})
    assert table._query_cache.length == 0
    assert len(table.search(q)) == 3


def test_search_result_list_is_a_copy_of_the_cache(table):
    q = where('age') > 26
    first = table.search(q)
    first.append('junk')
    second = table.search(q)
    assert len(second) == 2
    second.clear()
    assert len(table.search(q)) == 2
    table.clear_cache()
    assert table._query_cache.length == 0


def test_uncacheable_query_and_cache_size(db, table):
    q = Query().name.map(len) == 5
    assert [d['name'] for d in table.search(q)] == ['alice', 'carol']
    assert table._query_cache.length == 0

    small = db.table('small', cache_size=1)
    small.insert_multiple([{'a': 1}, {'a': 2}])
    small.search(where('a') == 1)
    small.search(where('a') == 2)
    assert small._query_cache.length == 1


def test_update_with_dict_and_cond(table):
    updated = table.update({'age': 31}, where('name') == 'alice')
    assert updated == [1]
    assert table.get(doc_id=1)['age'] == 31
    assert table.get(doc_id=2)['age'] == 25


def test_update_with_callable_and_cache_invalidation(table):
    q = where('age') == 26
    assert table.search(q) == []
    updated = table.update(increment('age'), where('age') < 31)
    assert sorted(updated) == [1, 2]
    assert table.get(doc_id=1)['age'] == 31
    assert table.get(doc_id=3)['age'] == 35
    assert [d.doc_id for d in table.search(q)] == [2]


def test_update_without_cond_and_by_doc_ids(table):
    assert table.update({'flag': True}) == [1, 2, 3]
    assert all(d['flag'] is True for d in table.all())
    updated = table.update({'extra': 1}, doc_ids=[2, 99])
    assert updated == [2]
    assert table.get(doc_id=2)['extra'] == 1
    assert 'extra' not in table.get(doc_id=1)


def test_update_multiple(table):
    updated = table.update_multiple([
        ({'tag': 'x'}, where('name') == 'alice'),
        (increment('age'), where('name') == 'bob'),
    ])
    assert updated == [1, 2]
    assert table.get(doc_id=1)['tag'] == 'x'
    assert table.get(doc_id=2)['age'] == 26


def test_upsert_updates_existing_or_inserts(table):
    ids = table.upsert({'name': 'alice', 'age': 40}, where('name') == 'alice')
    assert ids == [1]
    assert len(table) == 3
    assert table.get(doc_id=1)['age'] == 40

    ids = table.upsert({'name': 'erin', 'age': 22}, where('name') == 'erin')
    assert ids == [4]
    assert len(table) == 4
    assert table.get(doc_id=4)['name'] == 'erin'


def test_upsert_with_document_uses_doc_id(db):
    t = db.table('t')
    assert t.upsert(Document({'x': 1}, 7)) == [7]
    assert t.get(doc_id=7) == {'x': 1}
    assert t.upsert(Document({'y': 2}, 7)) == [7]
    assert t.get(doc_id=7) == {'x': 1, 'y': 2}


def test_upsert_without_cond_or_document_raises(db):
    t = db.table('t')
    with pytest.raises(ValueError):
        t.upsert({'x': 1})


def test_remove_by_cond(table):
    removed = table.remove(where('age') > 26)
    assert removed == [1, 3]
    assert [d['name'] for d in table.all()] == ['bob']


def test_remove_by_doc_ids_skips_missing_and_requires_arguments(table):
    removed = table.remove(doc_ids=[1, 99])
    assert removed == [1]
    assert len(table) == 2
    assert table.get(doc_id=1) is None
    with pytest.raises(RuntimeError):
        table.remove()
    assert len(table) == 2


def test_truncate_clears_documents_and_resets_ids(table):
    table.truncate()
    assert len(table) == 0
    assert table.all() == []
    assert table.insert({'a': 1}) == 1


def test_table_name_storage_repr_and_persist_empty(db):
    t = db.table('things')
    assert t.name == 'things'
    assert t.storage is db.storage
    assert repr(t).startswith("<Table name='things', total=0")
    db.table('visible', persist_empty=True)
    assert db.tables() == {'visible'}


def test_documents_in_different_tables_are_isolated(db):
    a = db.table('a')
    b = db.table('b')
    a.insert({'v': 1})
    assert len(a) == 1
    assert len(b) == 0
    assert b.insert({'v': 2}) == 1
    assert a.get(doc_id=1) == {'v': 1}
