import json

import pytest
from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage


@pytest.fixture
def db():
    return TinyDB(storage=MemoryStorage)


def test_insert_returns_incrementing_ids(db):
    assert db.insert({'a': 1}) == 1
    assert db.insert({'a': 2}) == 2
    assert db.insert({'a': 3}) == 3


def test_default_table_all_returns_inserted_documents(db):
    db.insert({'int': 1, 'char': 'a'})
    db.insert({'int': 1, 'char': 'b'})
    assert db.all() == [{'int': 1, 'char': 'a'}, {'int': 1, 'char': 'b'}]


def test_table_returns_same_instance_for_same_name(db):
    assert db.table('users') is db.table('users')


def test_tables_lists_names_of_populated_tables(db):
    db.table('a').insert({'x': 1})
    db.table('b').insert({'x': 2})
    assert db.tables() == {'a', 'b'}


def test_tables_are_isolated_from_each_other(db):
    first = db.table('first')
    second = db.table('second')
    first.insert({'v': 1})
    second.insert({'v': 2})
    second.insert({'v': 3})
    assert first.all() == [{'v': 1}]
    assert second.all() == [{'v': 2}, {'v': 3}]


def test_default_table_data_not_visible_in_named_table(db):
    db.insert({'x': 1})
    assert db.table('named').all() == []


def test_drop_table_removes_only_that_table(db):
    db.table('a').insert({'x': 1})
    db.table('b').insert({'x': 2})
    db.drop_table('a')
    assert db.tables() == {'b'}
    assert db.table('b').all() == [{'x': 2}]


def test_drop_tables_removes_everything(db):
    db.insert({'x': 1})
    db.table('a').insert({'x': 2})
    db.drop_tables()
    assert db.tables() == set()
    assert db.all() == []


def test_persist_empty_table_is_listed(db):
    db.table('kept', persist_empty=True)
    assert 'kept' in db.tables()


def test_context_manager_returns_database_and_persists(tmp_path):
    path = str(tmp_path / 'db.json')
    with TinyDB(path) as database:
        assert isinstance(database, TinyDB)
        database.insert({'foo': 'bar'})
    with TinyDB(path) as reopened:
        assert reopened.all() == [{'foo': 'bar'}]


def test_json_file_layout_uses_table_names_and_string_ids(tmp_path):
    path = tmp_path / 'db.json'
    with TinyDB(str(path)) as database:
        database.insert({'a': 1})
    with open(path) as handle:
        content = json.load(handle)
    assert content == {'_default': {'1': {'a': 1}}}
