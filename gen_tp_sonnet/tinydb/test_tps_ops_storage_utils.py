import json

import pytest
from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware, Middleware
from tinydb.operations import add, decrement, delete, increment, subtract
from tinydb.operations import set as set_field
from tinydb.storages import JSONStorage, MemoryStorage, Storage
from tinydb.utils import LRUCache, freeze


@pytest.fixture
def db():
    database = TinyDB(storage=MemoryStorage)
    database.insert({'name': 'a', 'count': 10})
    return database


def test_operation_delete_removes_field(db):
    db.update(delete('count'), where('name') == 'a')
    assert db.all() == [{'name': 'a'}]


def test_operation_add(db):
    db.update(add('count', 5), where('name') == 'a')
    assert db.get(doc_id=1)['count'] == 15


def test_operation_subtract(db):
    db.update(subtract('count', 4), where('name') == 'a')
    assert db.get(doc_id=1)['count'] == 6


def test_operation_set(db):
    db.update(set_field('count', 'many'), where('name') == 'a')
    assert db.get(doc_id=1)['count'] == 'many'


def test_operation_increment_and_decrement(db):
    db.update(increment('count'))
    assert db.get(doc_id=1)['count'] == 11
    db.update(decrement('count'))
    db.update(decrement('count'))
    assert db.get(doc_id=1)['count'] == 9


def test_memory_storage_write_then_read():
    storage = MemoryStorage()
    data = {'_default': {'1': {'a': 1}}}
    storage.write(data)
    assert storage.read() == data


def test_storage_is_abstract():
    with pytest.raises(TypeError):
        Storage()


def test_json_storage_creates_file_and_reads_none(tmp_path):
    path = tmp_path / 'new.json'
    storage = JSONStorage(str(path))
    try:
        assert path.exists()
        assert storage.read() is None
    finally:
        storage.close()


def test_json_storage_write_then_read(tmp_path):
    path = tmp_path / 'data.json'
    storage = JSONStorage(str(path))
    try:
        data = {'_default': {'1': {'a': 1, 'b': [1, 2]}}}
        storage.write(data)
        assert storage.read() == data
    finally:
        storage.close()


def test_json_storage_create_dirs(tmp_path):
    path = tmp_path / 'nested' / 'dirs' / 'db.json'
    storage = JSONStorage(str(path), create_dirs=True)
    storage.close()
    assert path.exists()


def test_caching_middleware_persists_on_close(tmp_path):
    path = str(tmp_path / 'cache.json')
    database = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    database.insert({'a': 1})
    assert database.all() == [{'a': 1}]
    database.close()
    with TinyDB(path) as reopened:
        assert reopened.all() == [{'a': 1}]


def test_caching_middleware_flush_writes_to_disk(tmp_path):
    path = tmp_path / 'flush.json'
    middleware = CachingMiddleware(JSONStorage)
    database = TinyDB(str(path), storage=middleware)
    database.insert({'a': 1})
    middleware.flush()
    assert json.loads(path.read_text()) == {'_default': {'1': {'a': 1}}}
    database.close()


def test_custom_middleware_sees_writes():
    class WriteCounter(Middleware):
        def __init__(self, storage_cls):
            super().__init__(storage_cls)
            self.writes = 0

        def write(self, data):
            self.writes += 1
            return self.storage.write(data)

    counter = WriteCounter(MemoryStorage)
    database = TinyDB(storage=counter)
    database.insert({'a': 1})
    database.insert({'a': 2})
    assert counter.writes >= 2
    assert database.all() == [{'a': 1}, {'a': 2}]


def test_lru_cache_evicts_least_recently_used():
    cache = LRUCache(capacity=2)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    assert 'a' not in cache
    assert 'b' in cache and 'c' in cache
    assert len(cache) == 2


def test_lru_cache_access_refreshes_entry():
    cache = LRUCache(capacity=2)
    cache['a'] = 1
    cache['b'] = 2
    assert cache.get('a') == 1
    cache['c'] = 3
    assert 'a' in cache
    assert 'b' not in cache


def test_freeze_makes_nested_structures_hashable():
    frozen = freeze({'a': [1, 2, {'b': 3}], 'c': {4, 5}})
    assert isinstance(hash(frozen), int)
    assert hash(frozen) == hash(freeze({'a': [1, 2, {'b': 3}], 'c': {4, 5}}))


def test_freeze_converts_list_to_tuple_and_set_to_frozenset():
    assert freeze([1, 2, 3]) == (1, 2, 3)
    assert freeze({1, 2}) == frozenset({1, 2})
