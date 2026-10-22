import json
import os

import pytest

from tinydb import JSONStorage, Storage, TinyDB, where
from tinydb.middlewares import CachingMiddleware, Middleware
from tinydb.storages import MemoryStorage


@pytest.fixture
def mem_db():
    return TinyDB(storage=MemoryStorage)


@pytest.fixture
def json_path(tmp_path):
    return str(tmp_path / 'db.json')


def test_default_table_receives_direct_calls(mem_db):
    assert mem_db.insert({'a': 1}) == 1
    assert len(mem_db) == 1
    assert [d['a'] for d in mem_db] == [1]
    assert mem_db.table('_default').all() == [{'a': 1}]
    assert mem_db.search(where('a') == 1) == [{'a': 1}]
    assert mem_db.name == '_default'


def test_table_returns_same_instance_for_same_name(mem_db):
    assert mem_db.table('x') is mem_db.table('x')
    assert mem_db.table('x') is not mem_db.table('y')
    assert mem_db.table('_default') is mem_db.table(TinyDB.default_table_name)


def test_tables_listing_and_custom_default_table_name(mem_db):
    assert mem_db.tables() == set()
    mem_db.insert({'a': 1})
    mem_db.table('other').insert({'b': 2})
    assert mem_db.tables() == {'_default', 'other'}

    class MyDB(TinyDB):
        default_table_name = 'custom'

    custom = MyDB(storage=MemoryStorage)
    custom.insert({'a': 1})
    assert custom.tables() == {'custom'}


def test_drop_table_removes_only_that_table(mem_db):
    mem_db.drop_table('missing')  # no-op on an empty database
    mem_db.table('a').insert({'x': 1})
    mem_db.table('b').insert({'x': 2})
    mem_db.drop_table('missing')  # no-op for unknown names
    mem_db.drop_table('a')
    assert mem_db.tables() == {'b'}
    assert len(mem_db.table('a')) == 0


def test_drop_tables_removes_everything(mem_db):
    mem_db.insert({'a': 1})
    mem_db.table('other').insert({'b': 2})
    mem_db.drop_tables()
    assert mem_db.tables() == set()
    assert len(mem_db) == 0
    assert mem_db.insert({'c': 3}) == 1


def test_repr_mentions_tables(mem_db):
    mem_db.insert({'a': 1})
    text = repr(mem_db)
    assert text.startswith('<TinyDB ')
    assert 'tables_count=1' in text
    assert 'default_table_documents_count=1' in text


def test_json_storage_persists_across_reopen_with_string_ids(json_path):
    with TinyDB(json_path) as db:
        db.insert({'a': 1})
        db.table('other').insert({'b': 2})
    with open(json_path) as f:
        assert json.load(f) == {'_default': {'1': {'a': 1}},
                                'other': {'1': {'b': 2}}}
    with TinyDB(json_path) as db:
        assert db.all() == [{'a': 1}]
        assert db.table('other').get(doc_id=1) == {'b': 2}


def test_context_manager_closes_storage_handle(json_path):
    with TinyDB(json_path) as db:
        db.insert({'a': 1})
    assert db.storage._handle.closed

    with TinyDB(json_path) as db2:
        db2.close()
    assert db2.storage._handle.closed


def test_json_storage_reads_existing_file(json_path):
    with open(json_path, 'w') as f:
        json.dump({'_default': {'5': {'k': 'v'}}}, f)
    with TinyDB(json_path) as db:
        assert db.get(doc_id=5) == {'k': 'v'}
        assert db.insert({'k': 'w'}) == 6


def test_json_storage_empty_file_reads_as_none_and_create_dirs(tmp_path, json_path):
    storage = JSONStorage(json_path)
    try:
        assert storage.read() is None
    finally:
        storage.close()

    nested = str(tmp_path / 'nested' / 'deeper' / 'db.json')
    with TinyDB(nested, create_dirs=True) as db:
        db.insert({'a': 1})
    assert os.path.exists(nested)


def test_json_storage_passes_kwargs_to_dumps(json_path):
    with TinyDB(json_path, indent=2) as db:
        db.insert({'a': 1})
    with open(json_path) as f:
        content = f.read()
    assert '\n  "_default"' in content


def test_json_file_shrinks_when_data_removed(json_path):
    with TinyDB(json_path) as db:
        db.insert({'payload': 'x' * 500})
        db.truncate()
    with open(json_path) as f:
        assert json.load(f) == {'_default': {}}


def test_json_storage_read_only_mode_rejects_writes(json_path):
    with open(json_path, 'w') as f:
        f.write('{}')
    db = TinyDB(json_path, access_mode='r')
    try:
        assert db.all() == []
        with pytest.raises(IOError):
            db.insert({'a': 1})
    finally:
        db.close()


def test_memory_storage_roundtrip_and_abstract_base():
    s = MemoryStorage()
    assert s.read() is None
    s.write({'t': {'1': {'a': 1}}})
    assert s.read() == {'t': {'1': {'a': 1}}}
    with pytest.raises(TypeError):
        Storage()


def test_custom_storage_class_is_used():
    class ListStorage(Storage):
        def __init__(self):
            self.data = None
            self.writes = 0

        def read(self):
            return self.data

        def write(self, data):
            self.writes += 1
            self.data = data

    db = TinyDB(storage=ListStorage)
    db.insert({'a': 1})
    db.insert({'a': 2})
    assert db.storage.writes == 2
    assert len(db) == 2


def test_middleware_call_returns_itself_and_builds_storage():
    mw = Middleware(MemoryStorage)
    assert mw.storage is None
    assert mw() is mw
    assert isinstance(mw.storage, MemoryStorage)


def test_caching_middleware_defers_writes_until_flush_or_cache_size():
    db = TinyDB(storage=CachingMiddleware(MemoryStorage))
    db.insert({'a': 1})
    assert len(db) == 1
    assert db.storage.storage.memory is None
    db.storage.flush()
    assert db.storage.storage.memory == {'_default': {'1': {'a': 1}}}

    class SmallCache(CachingMiddleware):
        WRITE_CACHE_SIZE = 2

    small = TinyDB(storage=SmallCache(MemoryStorage))
    small.insert({'a': 1})
    assert small.storage.storage.memory is None
    small.insert({'a': 2})
    assert len(small.storage.storage.memory['_default']) == 2


def test_caching_middleware_flushes_on_close(json_path):
    db = TinyDB(json_path, storage=CachingMiddleware(JSONStorage))
    db.insert({'a': 1})
    assert os.path.getsize(json_path) == 0
    db.close()
    with TinyDB(json_path) as reopened:
        assert reopened.all() == [{'a': 1}]
