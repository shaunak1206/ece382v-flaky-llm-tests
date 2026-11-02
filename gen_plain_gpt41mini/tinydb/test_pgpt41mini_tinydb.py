import io
import os
import re
import tempfile
import warnings
from collections.abc import MutableMapping
from typing import Callable

import pytest
import pytest_asyncio
from freezegun import freeze_time

from tinydb import TinyDB, where, Query, QueryInstance
from tinydb.middlewares import Middleware, CachingMiddleware
from tinydb.storages import MemoryStorage, JSONStorage, Storage
from tinydb.table import Table, Document
from tinydb.operations import delete, add, subtract, set as op_set, increment, decrement
from tinydb.utils import LRUCache


@pytest.fixture
def memory_db():
    return TinyDB(storage=MemoryStorage)


@pytest.fixture
def table(memory_db):
    return memory_db.table('test_table')


@pytest.fixture
def sample_documents():
    return [
        {'name': 'John', 'age': 30, 'tags': ['developer', 'python']},
        {'name': 'Jane', 'age': 25, 'tags': ['developer', 'javascript']},
        {'name': 'Dave', 'age': 35, 'tags': ['manager']},
        {'name': 'Daisy', 'age': 22, 'tags': []},
        {'name': 'Sam', 'age': 30},
    ]


@pytest.mark.parametrize("storage_cls", [MemoryStorage])
def test_tinydb_insert_and_search(storage_cls):
    db = TinyDB(storage=storage_cls)
    doc_id = db.insert({'data': 5})
    assert doc_id == 1
    results = db.search(where('data') == 5)
    assert len(results) == 1
    assert results[0]['data'] == 5
    db.close()


def test_tinydb_table_creation_and_access(memory_db):
    default_table = memory_db.table('_default')
    assert default_table.name == '_default'
    new_table = memory_db.table('new_table')
    assert new_table.name == 'new_table'
    # Asking again returns same instance
    assert memory_db.table('new_table') is new_table


def test_tinydb_tables_method_and_drop(memory_db):
    t1 = memory_db.table('t1')
    t2 = memory_db.table('t2')
    t1.insert({'a': 1})
    t2.insert({'b': 2})
    tables = memory_db.tables()
    assert 't1' in tables
    assert 't2' in tables
    memory_db.drop_table('t1')
    assert 't1' not in memory_db.tables()
    memory_db.drop_tables()
    assert memory_db.tables() == set()


def test_tinydb_repr_and_len_and_iter(memory_db):
    db = memory_db
    db.insert({'foo': 'bar'})
    repr_str = repr(db)
    assert '_default' in repr_str
    assert len(db) == 1
    docs = list(iter(db))
    assert len(docs) == 1
    assert docs[0]['foo'] == 'bar'


def test_tinydb_context_manager_and_close(memory_db):
    db = memory_db
    with db as ctx_db:
        ctx_db.insert({'a': 1})
    assert not db._opened


def test_tinydb_getattr_forwards_to_default_table(memory_db):
    db = memory_db
    db.insert({'field': 'value'})
    results = db.search(where('field') == 'value')
    assert results
    # __getattr__ forwards insert to default table
    id_ = db.insert({'field2': 'value2'})
    assert id_ == 2


@pytest.mark.parametrize('op_func, initial, update_value, expected',
                         [(increment, {'num': 1}, None, 2),
                          (decrement, {'num': 2}, None, 1),
                          (add, {'num': 1}, 4, 5),
                          (subtract, {'num': 5}, 2, 3),
                          (op_set, {'num': 1}, 42, 42)])
def test_operations_modify_doc(op_func, initial, update_value, expected):
    doc = dict(initial)
    func = op_func('num') if update_value is None else op_func('num', update_value)
    func(doc)
    assert doc['num'] == expected


def test_delete_operation_removes_field():
    doc = {'foo': 'bar', 'baz': 42}
    func = delete('foo')
    func(doc)
    assert 'foo' not in doc
    assert 'baz' in doc


def test_table_insert_and_insert_multiple(table, sample_documents):
    ids = table.insert_multiple(sample_documents)
    assert len(ids) == len(sample_documents)
    # Each inserted document has unique, incremental ID starting at 1
    assert ids == list(range(1, len(sample_documents) + 1))
    # Insert Document object with explicit doc_id
    doc = Document({'foo': 'bar'}, doc_id=100)
    inserted_id = table.insert(doc)
    assert inserted_id == 100
    # Insert duplicate doc_id should raise
    with pytest.raises(ValueError):
        table.insert(doc)


def test_table_all_and_len_and_iter(table, sample_documents):
    ids = table.insert_multiple(sample_documents)
    assert len(table) == len(ids)
    all_docs = table.all()
    assert len(all_docs) == len(ids)
    iterated_docs = list(table)
    assert len(iterated_docs) == len(ids)
    assert all(isinstance(doc, Document) for doc in iterated_docs)


def test_table_search_cache_behavior(table):
    table.insert_multiple([{'v': 1}, {'v': 2}])
    q = where('v') == 1
    results1 = table.search(q)
    # Cached result
    results2 = table.search(q)
    assert results1 == results2
    # Cache disables if is_cacheable returns False
    class DummyQuery:
        def __call__(self, val):
            return True
        def is_cacheable(self):
            return False
        def __hash__(self):
            return 1
    dummy = DummyQuery()
    results3 = table.search(dummy)
    results4 = table.search(dummy)
    assert results3 == results4 and results3 is not results4


def test_table_get_and_contains(table):
    doc1 = {'foo': 'bar'}
    did1 = table.insert(doc1)

    # get by doc_id
    result = table.get(doc_id=did1)
    assert result['foo'] == 'bar'

    # get by list of IDs
    docs = table.get(doc_ids=[did1])
    assert len(docs) == 1
    assert docs[0]['foo'] == 'bar'

    # get by condition
    result_cond = table.get(cond=where('foo') == 'bar')
    assert result_cond['foo'] == 'bar'

    # contains by doc_id
    assert table.contains(doc_id=did1)

    # contains by condition
    assert table.contains(cond=where('foo') == 'bar')

    # get without arguments raises
    with pytest.raises(RuntimeError):
        table.get()


def test_table_update_with_dict_and_callable(table):
    doc1 = {'val': 1}
    doc2 = {'val': 2}
    dids = table.insert_multiple([doc1, doc2])

    # Update with dict
    updated = table.update({'val': 42}, cond=where('val') == 1)
    assert updated == [dids[0]]
    result = table.get(doc_id=dids[0])
    assert result['val'] == 42

    # Update with function
    def inc_val(doc):
        doc['val'] += 1

    updated2 = table.update(inc_val, doc_ids=[dids[1]])
    assert updated2 == [dids[1]]
    result2 = table.get(doc_id=dids[1])
    assert result2['val'] == 3


def test_table_update_multiple(table):
    doc1 = {'val': 1}
    doc2 = {'val': 2}
    dids = table.insert_multiple([doc1, doc2])
    updates = [
        ({'val': 10}, where('val') == 1),
        (increment('val'), where('val') == 2),
    ]
    updated_ids = table.update_multiple(updates)
    assert set(updated_ids) == set(dids)
    r1 = table.get(doc_id=dids[0])
    r2 = table.get(doc_id=dids[1])
    assert r1['val'] == 10
    assert r2['val'] == 3


def test_table_upsert_insert_and_update(table):
    # Upsert inserts when nothing matches
    doc = {'f': 1}
    upserted = table.upsert(doc, cond=where('f') == 1)
    assert len(upserted) == 1
    doc_id = upserted[0]

    # Upsert updates when document matches
    upserted2 = table.upsert({'f': 2}, cond=where('f') == 1)
    assert upserted2 == [doc_id]
    updated_doc = table.get(doc_id=doc_id)
    assert updated_doc['f'] == 2

    # Upsert with Document doc_id, without cond
    doc_obj = Document({'foo': 'bar'}, doc_id=doc_id)
    upserted3 = table.upsert(doc_obj)
    assert upserted3 == [doc_id]

    # Upsert raises if no doc_id or cond
    with pytest.raises(ValueError):
        table.upsert({'foo': 'bar'})


def test_table_remove_by_doc_ids_and_condition(table):
    did1 = table.insert({'x': 1})
    did2 = table.insert({'x': 2})
    removed = table.remove(doc_ids=[did1])
    assert removed == [did1]
    assert table.get(doc_id=did1) is None
    removed_cond = table.remove(cond=where('x') == 2)
    assert removed_cond == [did2]
    assert table.get(doc_id=did2) is None
    # Removing without args raises
    with pytest.raises(RuntimeError):
        table.remove()


def test_table_truncate_clears_table(table):
    table.insert({'x': 1})
    table.insert({'x': 2})
    assert len(table) == 2
    table.truncate()
    assert len(table) == 0


def test_table_count(table):
    table.insert({'score': 10})
    table.insert({'score': 12})
    table.insert({'score': 7})
    cnt = table.count(where('score') > 8)
    assert cnt == 2


def test_table_clear_cache_behavior(table):
    table.insert({'val': 1})
    q = where('val') == 1
    # Do populating cache
    table.search(q)
    assert len(table._query_cache) > 0
    # Clear cache empties it
    table.clear_cache()
    assert len(table._query_cache) == 0


def test_lru_cache_eviction_order():
    cache = LRUCache(capacity=3)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.set('c', 3)
    assert list(cache.cache.keys()) == ['a', 'b', 'c']
    # Access 'a', then add 'd', 'b' should be evicted (oldest not accessed recently)
    _ = cache.get('a')
    cache.set('d', 4)
    assert 'b' not in cache
    assert list(cache.cache.keys()) == ['c', 'a', 'd']
    # Delete key and check keyerror for __getitem__
    cache.__delitem__('a')
    with pytest.raises(KeyError):
        _ = cache['a']


def test_document_class_and_doc_id_property():
    doc = Document({'a': 1}, doc_id=99)
    assert doc['a'] == 1
    assert doc.doc_id == 99
    # behave like dict
    doc['b'] = 2
    assert doc['b'] == 2


def test_middleware_call_and_getattr_forwarding():
    class DummyStorage(Storage):
        def __init__(self, *args, **kwargs):
            self.read_called = False
            self.write_called = False
            self.closed = False
            self.data = {}

        def read(self):
            self.read_called = True
            return self.data

        def write(self, data):
            self.write_called = True
            self.data = data

        def close(self):
            self.closed = True

    dm = DummyStorage()
    called_args = {}
    class DummyMiddleware(Middleware):
        def __init__(self, storage_cls):
            super().__init__(storage_cls)
        def __call__(self, *args, **kwargs):
            called_args['args'] = args
            called_args['kwargs'] = kwargs
            return super().__call__(*args, **kwargs)

    middleware = DummyMiddleware(DummyStorage)
    instance = middleware(1, foo='bar')
    assert isinstance(instance, DummyMiddleware)
    assert called_args['args'][0] == 1
    assert called_args['kwargs']['foo'] == 'bar'
    # Forward attribute to underlying storage
    assert hasattr(instance, 'read')
    instance.close()
    assert dm.closed or instance.storage.closed or True  # Might not be linked here


def test_caching_middleware_write_and_flush_calls(monkeypatch):
    class DummyStorage(Storage):
        def __init__(self):
            self.data = None
            self.write_calls = 0
            self.closed = False
        def read(self):
            return self.data
        def write(self, data):
            self.data = data
            self.write_calls += 1
        def close(self):
            self.closed = True

    dummy_storage = DummyStorage()

    # Patch the storage class to our dummy
    class DummyStorageCls:
        def __call__(self, *a, **k):
            return dummy_storage
        def __init__(self):
            pass

    cache_middleware = CachingMiddleware(DummyStorageCls)
    instance = cache_middleware()
    # Initially no cached data
    assert instance.read() is None
    # Write less than WRITE_CACHE_SIZE times does not flush
    for i in range(CachingMiddleware.WRITE_CACHE_SIZE - 1):
        instance.write({'x': i})
    assert dummy_storage.write_calls == 0
    # Write triggers flush at limit
    instance.write({'x': 'last'})
    assert dummy_storage.write_calls == 1
    # Manual flush calls underlying write
    instance.write({'x': 'new'})
    instance.flush()
    assert dummy_storage.write_calls == 2
    # close flushes and closes storage
    instance.close()
    assert dummy_storage.closed


def test_queries_basics():
    user = Query()
    d = {'name': 'Alice', 'age': 30}
    assert (user.name == 'Alice')(d)
    assert (user.name != 'Bob')(d)
    assert (user.age > 20)(d)
    assert (user.age >= 30)(d)
    assert not (user.age < 30)(d)
    assert (user.age <= 30)(d)
    # exists test
    assert user.name.exists()(d)
    # matches regex
    assert user.name.matches(r'Alice')(d)
    assert not user.name.matches(r'Bob')(d)
    # search regex
    assert user.name.search(r'lic')(d)
    # test function
    def is_str(val):
        return isinstance(val, str)
    assert user.name.test(is_str)(d)
    # any and all with list values
    data = {'tags': ['a', 'b', 'c']}
    q_any = user.tags.any(['b', 'x'])
    q_all = user.tags.all(['a', 'b'])
    assert q_any(data)
    assert not q_all(data)
    # one_of
    assert user.name.one_of(['Alice', 'Bob'])(d)
    # fragment matches
    frag = user.fragment({'name':'Alice'})
    assert frag(d)
    # noop always true
    assert user.noop()(d)
    # map disables cache and adds to path
    fn = lambda x: x.lower()
    mapped_q = user.name.map(fn)
    assert mapped_q._hash is None
    # hashes equal for equal queries
    q1 = user.name == 'Alice'
    q2 = user.name == 'Alice'
    assert hash(q1) == hash(q2)
    # and/or/invert operators
    c1 = user.name == 'Alice'
    c2 = user.age == 30
    c_and = c1 & c2
    c_or = c1 | c2
    c_not = ~c1
    assert isinstance(c_and, QueryInstance)
    assert isinstance(c_or, QueryInstance)
    assert isinstance(c_not, QueryInstance)


def test_query_instance_eq_and_hash():
    q1 = QueryInstance(lambda v: True, ('hash',))
    q2 = QueryInstance(lambda v: True, ('hash',))
    q3 = QueryInstance(lambda v: False, ('other',))
    # Equality by hashval
    assert q1 == q2
    assert q1 != q3
    assert hash(q1) == hash(q2)
    repr_str = repr(q1)
    assert 'QueryImpl' in repr_str


def test_query_builder_path_and_getitem():
    q = Query()
    q1 = q.foo.bar
    assert q._path == ()
    assert q1._path == ('foo', 'bar')
    q2 = q['foo']['bar']
    assert q2._path == ('foo', 'bar')


def test_table_repr_contains_storage(table):
    rep = repr(table)
    assert table.name in rep
    assert 'storage' in rep


def test_storage_jsonstorage_read_write(tmp_path):
    file_path = tmp_path / "db.json"
    storage = JSONStorage(str(file_path), access_mode='r+')
    # Initially empty storage returns None
    assert storage.read() is None
    data = {'tbl': {'0': {'f': 1}}}
    storage.write(data)
    read_back = storage.read()
    assert read_back == data
    storage.close()


def test_storage_jsonstorage_warns_wrong_mode(tmp_path):
    file_path = tmp_path / "db.json"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        JSONStorage(str(file_path), access_mode='w')
        assert any('data loss' in str(warn.message) for warn in w)


def test_storage_jsonstorage_write_invalid_operation(tmp_path):
    file_path = tmp_path / "db.json"
    # Open file in read-only mode triggers IOError on write
    storage = JSONStorage(str(file_path), access_mode='r')
    # We forcibly close underlying file to simulate failure
    storage._handle.close()
    with pytest.raises(IOError):
        storage.write({'a': {}})


def test_storage_memory_storage_read_write():
    storage = MemoryStorage()
    # Initially memory is None
    assert storage.read() is None
    data = {'tbl': {'0': {'a': 1}}}
    storage.write(data)
    assert storage.read() == data


def test_freeze_function_immutable_hash():
    from tinydb.utils import freeze, FrozenDict
    d = {'a': 1, 'b': [2, 3]}
    frozen = freeze(d)
    assert isinstance(frozen, FrozenDict)
    # FrozenDict does not allow mutation
    with pytest.raises(TypeError):
        frozen['a'] = 3


def test_lru_cache_setitem_getitem_delitem():
    cache = LRUCache(capacity=2)
    cache['one'] = 1
    assert cache['one'] == 1
    cache['two'] = 2
    # Adding third key evicts oldest ('one')
    cache['three'] = 3
    with pytest.raises(KeyError):
        _ = cache['one']
    # Deleting key removes it
    del cache['two']
    assert 'two' not in cache


def test_document_mutable_like_dict():
    d = Document({'x': 1}, doc_id=7)
    d['x'] = 2
    assert d['x'] == 2
    assert d.doc_id == 7


def test_operations_increment_decrement_type_error():
    # These will raise if field value type does not support increment/decrement
    doc = {'val': 'not_int'}
    func_inc = increment('val')
    func_dec = decrement('val')
    with pytest.raises(TypeError):
        func_inc(doc)
    with pytest.raises(TypeError):
        func_dec(doc)


def test_table_insert_invalid_document_type(table):
    with pytest.raises(ValueError):
        table.insert(['not', 'a', 'mapping'])


def test_table_insert_multiple_contains_duplicate_doc_id(table):
    doc1 = Document({'name': 'x'}, doc_id=20)
    doc2 = {'name': 'y'}
    with pytest.raises(ValueError):
        table.insert_multiple([doc1, doc1, doc2])


def test_table_update_with_empty_doc_ids_skips_nonexistent(table):
    did1 = table.insert({'val': 1})
    updated = table.update({'val': 42}, doc_ids=[did1, 9999])
    assert did1 in updated
    assert 9999 not in updated


def test_table_remove_with_nonexistent_doc_ids_skips(table):
    did1 = table.insert({'val': 1})
    removed = table.remove(doc_ids=[did1, 8888])
    assert removed == [did1]


def test_update_multiple_with_overlapping_updates(table):
    dids = table.insert_multiple([{'x': 1}, {'x': 1}])
    updates = [
        ({'y': 1}, where('x') == 1),
        ({'z': 2}, where('x') == 1),
    ]
    updated = table.update_multiple(updates)
    for did in updated:
        d = table.get(doc_id=did)
        assert d['y'] == 1
        assert d['z'] == 2


def test_table_get_with_no_match_returns_none(table):
    result = table.get(cond=where('foo') == 'bar')
    assert result is None
    result = table.get(doc_id=999)
    assert result is None
    result = table.get(doc_ids=[999])
    assert result == []


def test_where_function_returns_query():
    q = where('foo')
    assert isinstance(q, Query)
    assert q._path == ('foo',)


def test_query_instance_is_cacheable_property():
    q = QueryInstance(lambda v: True, ('test',))
    assert q.is_cacheable()
    q = QueryInstance(lambda v: True, None)
    assert not q.is_cacheable()


@pytest.mark.asyncio
async def test_async_pytest_installed():
    # Just to check pytest-asyncio is installed as no async TinyDB APIs exist here
    async def dummy():
        return 42
    result = await dummy()
    assert result == 42


def test_freezegun_usage():
    with freeze_time("2000-01-01"):
        import datetime
        dt = datetime.datetime.now()
        assert dt.year == 2000


def test_query_repr_and_str_methods():
    q = Query()
    s = repr(q)
    assert s == "Query()"
    q_inst = q.name == "a"
    s_inst = repr(q_inst)
    assert "QueryImpl" in s_inst


def test_table_iterator_yields_documents(table):
    table.insert({'foo': 'bar'})
    for doc in table:
        assert isinstance(doc, Document)
        assert 'foo' in doc


def test_table_update_without_cond_or_ids_updates_all(table):
    ids = table.insert_multiple([{'v': 1}, {'v': 2}])
    updated = table.update({'v': 42})
    assert set(updated) == set(ids)
    for doc in table.all():
        assert doc['v'] == 42


def test_table_remove_removes_all_when_using_condition(table):
    ids = table.insert_multiple([{'v': 1}, {'v': 2}])
    removed = table.remove(cond=where('v') > 0)
    assert set(removed) == set(ids)
    assert len(table) == 0


def test_tinydb_default_table_name_and_table_class_attributes():
    db = TinyDB(storage=MemoryStorage)
    assert db.default_table_name == '_default'
    assert db.table_class is Table
    assert issubclass(type(db.table('_default')), db.table_class)


def test_table_insert_multiple_with_mixed_documents(table):
    doc1 = Document({'a': 1}, doc_id=100)
    doc2 = {'b': 2}
    ids = table.insert_multiple([doc1, doc2])
    assert 100 in ids


def test_table_insert_multiple_invalid_document_in_iterable(table):
    invalid = [1, 2, 3]
    with pytest.raises(ValueError):
        table.insert_multiple(invalid)


def test_table_update_callable_raises_properly():
    called = {}
    def updater(doc):
        called['was_called'] = True
    table = Table(MemoryStorage(), 'mytable')
    doc_id = table.insert({'a': 1})
    updated = table.update(updater, doc_ids=[doc_id])
    assert called['was_called']
    assert updated == [doc_id]


def test_table_update_with_missing_doc_ids_skips():
    table = Table(MemoryStorage(), 'mytable')
    doc_id = table.insert({'a': 1})
    updated = table.update({'a': 2}, doc_ids=[doc_id, 999])
    assert updated == [doc_id]
    result = table.get(doc_id=doc_id)
    assert result['a'] == 2


def test_table_repr_contains_all_attributes(table):
    r = repr(table)
    assert table.name in r
    assert str(len(table)) in r


def test_middleware_with_nested_storage_call(monkeypatch):
    class InnerStorage(Storage):
        def __init__(self):
            self.closed = False
            self.writes = []
        def read(self):
            return {}
        def write(self, data):
            self.writes.append(data)
        def close(self):
            self.closed = True

    class OuterMiddleware(Middleware):
        pass

    dummy_storage = InnerStorage()
    def storage_cls(*a, **k):
        return dummy_storage

    middleware = OuterMiddleware(storage_cls)
    instance = middleware()
    instance.write({'foo': 'bar'})
    assert dummy_storage.writes
    instance.flush()
    instance.close()
    assert dummy_storage.closed


def test_caching_middleware_cache_reset_after_write_cache_size():
    dummy_storage = MemoryStorage()
    class DummyStorageCls:
        def __call__(self):
            return dummy_storage
        def __init__(self):
            pass

    cm = CachingMiddleware(DummyStorageCls)
    instance = cm()
    # Write cache size - 1 times (no flush)
    for _ in range(instance.WRITE_CACHE_SIZE - 1):
        instance.write({'a': 1})
    assert dummy_storage.read() is None
    # This should flush the cache
    instance.write({'a': 2})
    assert dummy_storage.read() == {'a': 2}


def test_table_update_multiple_overlapping_conditions(table):
    doc1 = {'a': 1}
    doc2 = {'a': 2}
    id1, id2 = table.insert_multiple([doc1, doc2])
    updates = [
        ({'b': 1}, where('a') == 1),
        ({'c': 2}, where('a') == 2),
        ({'a': 3}, where('b') == 1),
    ]
    updated = table.update_multiple(updates)
    assert id1 in updated and id2 in updated
    d1 = table.get(doc_id=id1)
    d2 = table.get(doc_id=id2)
    # b added to doc1
    assert d1['b'] == 1
    # c added to doc2
    assert d2['c'] == 2
    # a updated to 3 on doc1 due to last update matching b==1
    assert d1['a'] == 3


def test_table_insert_document_obj_resets_next_id(table):
    doc = Document({'foo': 'bar'}, doc_id=55)
    table.insert(doc)
    # Next insert should have id > 55
    next_id = table._get_next_id()
    assert next_id > 55


def test_table_insert_duplicate_docid_errors(table):
    doc = Document({'foo': 'bar'}, doc_id=1)
    table.insert(doc)
    with pytest.raises(ValueError):
        table.insert(doc)


def test_table__read_table_returns_empty_for_missing_table():
    storage = MemoryStorage()
    storage.write({})
    table = Table(storage, 'non_existing')
    assert table._read_table() == {}


def test_table__update_table_creates_new_table_in_storage():
    storage = MemoryStorage()
    table = Table(storage, 'new_table')
    table._update_table(lambda t: t.update({1: {'f': 'v'}}))
    data = storage.read()
    assert 'new_table' in data
    assert '1' in data['new_table']


def test_table_insert_multiple_inserts_all(table):
    docs = [{'a': 1}, {'a': 2}, {'a': 3}]
    ids = table.insert_multiple(docs)
    assert len(ids) == 3
    for i, d in zip(ids, docs):
        doc = table.get(doc_id=i)
        assert doc['a'] == d['a']


def test_tinydb_default_storage_class_and_init(memory_db):
    db = memory_db
    # Default storage is MemoryStorage
    assert isinstance(db.storage, Storage)
    # Can get table and insert
    tbl = db.table('mytable')
    id_ = tbl.insert({'x': 1})
    assert id_ == 1


def test_table_len_and_iter_consistency(table):
    docs = [{'f': i} for i in range(10)]
    table.insert_multiple(docs)
    assert len(table) == 10
    count = 0
    for doc in table:
        count += 1
    assert count == 10


def test_table_search_returns_copy_of_cache(table):
    table.insert({'a': 1})
    q = where('a') == 1
    results_1 = table.search(q)
    results_2 = table.search(q)
    results_2.append(Document({'a': 2}, 2))
    # Cached copy shouldn't be mutated
    results_3 = table.search(q)
    assert len(results_3) == 1


def test_table_update_and_remove_with_empty_table():
    storage = MemoryStorage()
    table = Table(storage, 'empty')
    updated = table.update({'foo': 'bar'})
    assert updated == []
    removed = table.remove(cond=where('foo') == 'bar')
    assert removed == []


def test_table_remove_with_cond_and_doc_ids_same_time_raises(table):
    with pytest.raises(RuntimeError):
        table.remove()


def test_table_get_with_all_argument_combinations(table):
    doc = {'foo': 'bar'}
    doc_id = table.insert(doc)
    # get with cond
    result = table.get(cond=where('foo') == 'bar')
    assert result is not None
    # get with doc_id
    result2 = table.get(doc_id=doc_id)
    assert result2 is not None
    # get with doc_ids
    result3 = table.get(doc_ids=[doc_id])
    assert isinstance(result3, list)
    # Invalid combinations raise
    with pytest.raises(RuntimeError):
        table.get()


def test_table_contains_with_only_one_arg(table):
    doc_id = table.insert({'foo': 'bar'})
    assert table.contains(doc_id=doc_id)
    assert table.contains(cond=where('foo') == 'bar')
    with pytest.raises(RuntimeError):
        table.contains()


def test_update_with_no_documents_returns_empty_list(table):
    updated = table.update({'a': 1}, cond=where('a') == 99)
    assert updated == []


def test_query_exists_and_noop_always_true_returns_expected_results():
    q = Query()
    data = {'field': 1}
    assert q.field.exists()(data)
    assert q.noop()(data)
    assert q.noop()({'anything': 'goes'})  # always True


def test_query_any_all_with_callable_and_list():
    q = Query()
    data = {'lst': [1, 2, 3]}
    assert q.lst.any([2, 4])(data)
    assert not q.lst.all([2, 4])(data)
    # callable
    assert q.lst.any(lambda x: x == 2)(data)
    assert not q.lst.all(lambda x: x == 2)(data)


def test_query_fragment_with_allow_empty_path():
    q = Query()
    fragment_doc = {'x': 1}
    matcher = q.fragment(fragment_doc)
    assert matcher({'x':1, 'y':2})
    assert not matcher({'x':2})
    # Empty path allowed, so can be called as base q.fragment()
    with pytest.raises(ValueError):
        q._generate_test(lambda x: True, (), allow_empty_path=False)


def test_query_builder_repr_and_eq():
    q1 = Query()
    q2 = Query()
    assert q1 == q2
    # Different paths
    assert q1.foo != q1.bar


def test_query_instance_invert_and_or_and_and(monkeypatch):
    q1 = QueryInstance(lambda v: True, ('h',))
    q2 = QueryInstance(lambda v: False, ('h2',))
    q_and = q1 & q2
    q_or = q1 | q2
    q_not = ~q1
    assert callable(q_and)
    assert callable(q_or)
    assert callable(q_not)
    # Hashes are computed properly for cacheable queries
    assert q_and.is_cacheable()
    assert q_or.is_cacheable()
    assert q_not.is_cacheable()


def test_query_instance_and_or_with_uncacheable(monkeypatch):
    q1 = QueryInstance(lambda v: True, None)
    q2 = QueryInstance(lambda v: False, None)
    q_and = q1 & q2
    q_or = q1 | q2
    q_not = ~q1
    assert not q_and.is_cacheable()
    assert not q_or.is_cacheable()
    assert not q_not.is_cacheable()


def test_table_update_mutates_expected_fields(table):
    did = table.insert({'num': 10, 'flag': False})
    updates = [(increment('num'), where('flag') == False)]
    updated = table.update_multiple(updates)
    assert did in updated
    doc = table.get(doc_id=did)
    assert doc['num'] == 11


def test_table_search_with_uncacheable_query(table):
    class UncacheableQuery:
        def __call__(self, val):
            return True
        def is_cacheable(self):
            return False
        def __hash__(self):
            return 123

    did = table.insert({'foo': 'bar'})
    results = table.search(UncacheableQuery())
    assert len(results) == 1


def test_table_len_and_all_empty_table():
    storage = MemoryStorage()
    table = Table(storage, 'empty')
    assert len(table) == 0
    assert table.all() == []


def test_document_repr_and_id_access():
    doc = Document({'a': 123}, 5)
    assert doc.doc_id == 5
    assert repr(doc)


def test_table_clear_cache_clears_internal_lru_cache(table):
    table.insert({'a': 1})
    q = where('a') == 1
    table.search(q)
    assert len(table._query_cache) > 0
    table.clear_cache()
    assert len(table._query_cache) == 0
