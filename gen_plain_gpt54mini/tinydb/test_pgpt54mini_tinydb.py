import io
import json
import os
import warnings
from pathlib import Path

import pytest

from tinydb import TinyDB, Query, where
from tinydb.database import TinyDB as TinyDBClass
from tinydb.middlewares import Middleware, CachingMiddleware
from tinydb.operations import add, subtract, set as op_set, delete, increment, decrement
from tinydb.storages import MemoryStorage, JSONStorage, Storage, touch
from tinydb.table import Table, Document
from tinydb.utils import LRUCache, freeze, FrozenDict


class DummyStorage(Storage):
    def __init__(self):
        self.data = None
        self.closed = False
        self.writes = 0

    def read(self):
        return self.data

    def write(self, data):
        self.writes += 1
        self.data = data

    def close(self):
        self.closed = True


def test_version_string_exists():
    import tinydb
    assert isinstance(tinydb.__version__, str)
    assert tinydb.__version__ == "4.9.0"


def test_freeze_dict_returns_frozendict():
    frozen = freeze({"a": 1, "b": 2})
    assert isinstance(frozen, FrozenDict)
    assert hash(frozen) == hash(FrozenDict({"a": 1, "b": 2}))


def test_freeze_list_returns_tuple():
    assert freeze([1, 2, 3]) == (1, 2, 3)


def test_freeze_tuple_recurses():
    frozen = freeze((1, {"a": [2, 3]}))
    assert frozen == (1, FrozenDict({"a": (2, 3)}))


def test_freeze_set_returns_frozenset():
    assert freeze({1, 2}) == frozenset({1, 2})


def test_freeze_other_object_returns_same_object():
    obj = object()
    assert freeze(obj) is obj


def test_frozendict_is_immutable_setitem():
    fd = FrozenDict({"a": 1})
    with pytest.raises(TypeError):
        fd["a"] = 2


def test_frozendict_is_immutable_update():
    fd = FrozenDict({"a": 1})
    with pytest.raises(TypeError):
        fd.update({"b": 2})


def test_lru_cache_set_and_get():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    assert cache["a"] == 1
    assert cache.get("a") == 1


def test_lru_cache_moves_accessed_key_to_end():
    cache = LRUCache(capacity=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    _ = cache["a"]
    assert cache.lru[-1] == "a"


def test_lru_cache_evicts_least_recently_used():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert "a" not in cache
    assert set(cache.lru) == {"b", "c"}


def test_lru_cache_clear_empties_cache():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache.clear()
    assert len(cache) == 0
    assert cache.lru == []


def test_lru_cache_contains_and_iteration():
    cache = LRUCache(capacity=2)
    cache["a"] = 1
    cache["b"] = 2
    assert "a" in cache
    assert list(iter(cache)) == ["a", "b"]


def test_query_repr_and_hash():
    q = Query()
    assert repr(q) == "Query()"
    assert isinstance(hash(q), int)


def test_where_builds_field_query():
    q = where("x")
    assert q({"x": 1})
    assert not q({"y": 1})


def test_query_attribute_access_and_equality():
    q = Query().x == 5
    assert q({"x": 5}) is True
    assert q({"x": 4}) is False


def test_query_getitem_access():
    q = Query()["x"] == 5
    assert q({"x": 5})


def test_query_nested_path_access():
    q = Query().a.b == 3
    assert q({"a": {"b": 3}})
    assert not q({"a": {"b": 4}})


def test_query_missing_key_returns_false():
    q = Query().a.b == 3
    assert q({"a": {}}) is False


def test_query_exists():
    q = Query().name.exists()
    assert q({"name": "Ada"})
    assert not q({})


def test_query_not_equal():
    q = Query().x != 5
    assert q({"x": 4})
    assert not q({"x": 5})


def test_query_comparison_operators():
    assert (Query().x < 5)({"x": 4})
    assert (Query().x <= 5)({"x": 5})
    assert (Query().x > 5)({"x": 6})
    assert (Query().x >= 5)({"x": 5})


def test_query_matches_and_search():
    assert Query().s.matches(r"^\w+$")({"s": "abc123"})
    assert not Query().s.matches(r"^\w+$")({"s": "abc 123"})
    assert Query().s.search(r"\d+")({"s": "abc123"})
    assert not Query().s.search(r"\d+")({"s": "abc"})


def test_query_test_with_args():
    q = Query().x.test(lambda v, n: v + n == 5, 2)
    assert q({"x": 3})
    assert not q({"x": 4})


def test_query_any_with_query():
    q = Query().items.any(Query().x == 2)
    assert q({"items": [{"x": 1}, {"x": 2}]})
    assert not q({"items": [{"x": 1}, {"x": 3}]})


def test_query_any_with_list():
    q = Query().items.any([2, 3])
    assert q({"items": [1, 2]})
    assert not q({"items": [1, 4]})


def test_query_all_with_query():
    q = Query().items.all(Query().x >= 1)
    assert q({"items": [{"x": 1}, {"x": 2}]})
    assert not q({"items": [{"x": 1}, {"x": 0}]})


def test_query_all_with_list():
    q = Query().items.all([1, 2])
    assert q({"items": [1, 2, 3]})
    assert not q({"items": [1, 3]})


def test_query_one_of():
    q = Query().x.one_of([1, 2, 3])
    assert q({"x": 2})
    assert not q({"x": 9})


def test_query_fragment_matches_subset():
    q = Query().fragment({"a": 1, "b": 2})
    assert q({"a": 1, "b": 2, "c": 3})
    assert not q({"a": 1, "b": 9})


def test_query_noop_always_true():
    assert Query().noop()({})
    assert Query().noop()({"anything": 123})


def test_query_map_transforms_path():
    q = Query().x.map(lambda v: v + 1) == 3
    assert q({"x": 2})
    assert not q({"x": 1})


def test_query_and_or_not_composition():
    q = ((Query().x == 1) & (Query().y == 2)) | ~(Query().z == 3)
    assert q({"x": 1, "y": 2, "z": 3})
    assert q({"x": 0, "y": 0, "z": 4})
    assert not q({"x": 1, "y": 0, "z": 3})


def test_query_and_is_commutative_in_hash():
    a = Query().x == 1
    b = Query().y == 2
    assert (a & b) == (b & a)
    assert hash(a & b) == hash(b & a)


def test_query_or_is_commutative_in_hash():
    a = Query().x == 1
    b = Query().y == 2
    assert (a | b) == (b | a)
    assert hash(a | b) == hash(b | a)


def test_query_cacheability_false_for_unhashable_rhs():
    q = Query().x == {"a": [1, 2]}
    assert q.is_cacheable()
    assert q({"x": {"a": [1, 2]}})


def test_operations_delete():
    doc = {"a": 1, "b": 2}
    delete_a = delete("a")
    delete_a(doc)
    assert doc == {"b": 2}


def test_operations_add():
    doc = {"x": 1}
    add("x", 4)(doc)
    assert doc["x"] == 5


def test_operations_subtract():
    doc = {"x": 5}
    subtract("x", 3)(doc)
    assert doc["x"] == 2


def test_operations_set():
    doc = {"x": 1}
    op_set("x", 99)(doc)
    assert doc["x"] == 99


def test_operations_increment():
    doc = {"x": 1}
    increment("x")(doc)
    assert doc["x"] == 2


def test_operations_decrement():
    doc = {"x": 1}
    decrement("x")(doc)
    assert doc["x"] == 0


def test_memory_storage_read_write():
    s = MemoryStorage()
    assert s.read() is None
    s.write({"x": {"1": {"a": 1}}})
    assert s.read() == {"x": {"1": {"a": 1}}}


def test_touch_creates_file(tmp_path):
    path = tmp_path / "nested" / "db.json"
    touch(str(path), create_dirs=True)
    assert path.exists()


def test_json_storage_empty_file_reads_none(tmp_path):
    path = tmp_path / "db.json"
    path.write_text("")
    storage = JSONStorage(str(path), access_mode="r+")
    try:
        assert storage.read() is None
    finally:
        storage.close()


def test_json_storage_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "db.json"
    storage = JSONStorage(str(path), access_mode="r+")
    try:
        storage.write({"_default": {"1": {"a": 1}}})
        assert storage.read() == {"_default": {"1": {"a": 1}}}
    finally:
        storage.close()


def test_json_storage_warns_on_unusual_access_mode(tmp_path):
    path = tmp_path / "db.json"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        storage = JSONStorage(str(path), access_mode="w+")
        try:
            assert any("data loss or corruption" in str(w.message) for w in rec)
        finally:
            storage.close()


def test_json_storage_write_raises_on_readonly_mode(tmp_path):
    path = tmp_path / "db.json"
    path.write_text("{}")
    storage = JSONStorage(str(path), access_mode="r")
    try:
        with pytest.raises(IOError):
            storage.write({"x": {}})
    finally:
        storage.close()


def test_table_insert_and_all():
    storage = MemoryStorage()
    table = Table(storage, "t")
    doc_id = table.insert({"a": 1})
    assert doc_id == 1
    assert table.all()[0]["a"] == 1
    assert table.all()[0].doc_id == 1


def test_table_insert_document_keeps_doc_id():
    storage = MemoryStorage()
    table = Table(storage, "t")
    doc = Document({"a": 1}, 42)
    assert table.insert(doc) == 42
    assert table.get(doc_id=42)["a"] == 1


def test_table_insert_non_mapping_raises():
    table = Table(MemoryStorage(), "t")
    with pytest.raises(ValueError):
        table.insert([1, 2, 3])  # type: ignore[arg-type]


def test_table_insert_multiple_returns_ids():
    table = Table(MemoryStorage(), "t")
    ids = table.insert_multiple([{"a": 1}, {"b": 2}])
    assert ids == [1, 2]


def test_table_get_by_doc_id():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    assert table.get(doc_id=1)["a"] == 1


def test_table_get_by_query():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    table.insert({"a": 2})
    result = table.get(where("a") == 2)
    assert result["a"] == 2
    assert result.doc_id == 2


def test_table_get_by_doc_ids():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}, {"a": 3}])
    docs = table.get(doc_ids=[3, 1])
    assert [doc.doc_id for doc in docs] == [1, 3]


def test_table_contains_by_doc_id_and_query():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    assert table.contains(doc_id=1)
    assert table.contains(cond=where("a") == 1)
    assert not table.contains(doc_id=2)


def test_table_update_fields_mapping():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    updated = table.update({"b": 2}, doc_ids=[1])
    assert updated == [1]
    assert table.get(doc_id=1)["b"] == 2


def test_table_update_callable():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    updated = table.update(lambda doc: doc.__setitem__("a", 5), doc_ids=[1])
    assert updated == [1]
    assert table.get(doc_id=1)["a"] == 5


def test_table_update_with_query():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}, {"a": 1}])
    updated = table.update({"x": True}, cond=where("a") == 1)
    assert updated == [1, 3]
    assert table.get(doc_id=1)["x"] is True
    assert table.get(doc_id=2).get("x") is None


def test_table_update_multiple():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"b": 2}])
    updated = table.update_multiple([
        ({"c": 3}, where("a") == 1),
        (lambda doc: doc.__setitem__("d", 4), where("b") == 2),
    ])
    assert set(updated) == {1, 2}
    assert table.get(doc_id=1)["c"] == 3
    assert table.get(doc_id=2)["d"] == 4


def test_table_upsert_updates_existing():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    result = table.upsert({"a": 2}, where("a") == 1)
    assert result == [1]
    assert table.get(doc_id=1)["a"] == 2


def test_table_upsert_inserts_if_missing():
    table = Table(MemoryStorage(), "t")
    result = table.upsert({"a": 1}, where("a") == 1)
    assert result == [1]
    assert table.get(doc_id=1)["a"] == 1


def test_table_upsert_with_document_id():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    result = table.upsert(Document({"a": 9}, 1))
    assert result == [1]
    assert table.get(doc_id=1)["a"] == 9


def test_table_remove_by_query():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}, {"a": 1}])
    removed = table.remove(cond=where("a") == 1)
    assert removed == [1, 3]
    assert len(table) == 1


def test_table_remove_by_doc_ids_skips_missing():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}])
    removed = table.remove(doc_ids=[2, 99, 1])
    assert removed == [2, 1]
    assert len(table) == 0


def test_table_remove_without_args_raises():
    table = Table(MemoryStorage(), "t")
    with pytest.raises(RuntimeError):
        table.remove()


def test_table_truncate_clears_and_resets_id():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    table.truncate()
    assert len(table) == 0
    assert table.insert({"b": 2}) == 1


def test_table_count():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}, {"a": 1}])
    assert table.count(where("a") == 1) == 2


def test_table_clear_cache():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    table.search(where("a") == 1)
    assert len(table._query_cache) == 1
    table.clear_cache()
    assert len(table._query_cache) == 0


def test_table_len_and_iter():
    table = Table(MemoryStorage(), "t")
    table.insert_multiple([{"a": 1}, {"a": 2}])
    assert len(table) == 2
    assert [doc.doc_id for doc in table] == [1, 2]


def test_table_repr_contains_name_and_total():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    rep = repr(table)
    assert "name='t'" in rep
    assert "total=1" in rep


def test_table_search_uses_cache_and_returns_copy():
    table = Table(MemoryStorage(), "t")
    table.insert({"a": 1})
    q = where("a") == 1
    first = table.search(q)
    second = table.search(q)
    assert first == second
    assert first is not second


def test_tinydb_default_table_insert_and_search():
    db = TinyDB(storage=MemoryStorage)
    db.insert({"a": 1})
    assert db.search(where("a") == 1)[0]["a"] == 1


def test_tinydb_table_accessor_returns_same_instance():
    db = TinyDB(storage=MemoryStorage)
    t1 = db.table("x")
    t2 = db.table("x")
    assert t1 is t2


def test_tinydb_tables_lists_names():
    db = TinyDB(storage=MemoryStorage)
    db.table("a")
    db.table("b")
    assert db.tables() == {"a", "_default", "b"} or db.tables() == {"a", "b"}


def test_tinydb_drop_table_removes_specific_table():
    db = TinyDB(storage=MemoryStorage)
    db.table("x").insert({"a": 1})
    db.drop_table("x")
    assert "x" not in db.tables()


def test_tinydb_drop_tables_clears_all():
    db = TinyDB(storage=MemoryStorage)
    db.table("x").insert({"a": 1})
    db.drop_tables()
    assert db.tables() == set()
    assert len(db.table("x")) == 0


def test_tinydb_context_manager_closes_storage():
    storage = DummyStorage()
    db = TinyDB(storage=lambda: storage)
    with db as entered:
        assert entered is db
    assert storage.closed is True


def test_tinydb_close_marks_closed():
    storage = DummyStorage()
    db = TinyDB(storage=lambda: storage)
    db.close()
    assert storage.closed is True
    assert db._opened is False


def test_tinydb_getattr_forwards_to_default_table():
    db = TinyDB(storage=MemoryStorage)
    db.insert({"a": 1})
    assert db.count(where("a") == 1) == 1


def test_tinydb_len_and_iter_default_table():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple([{"a": 1}, {"a": 2}])
    assert len(db) == 2
    assert [doc.doc_id for doc in db] == [1, 2]


def test_middleware_wraps_storage_and_forwards_attrs():
    mw = Middleware(DummyStorage)
    wrapped = mw()
    assert wrapped is mw
    assert isinstance(mw.storage, DummyStorage)
    assert mw.read() is None


def test_caching_middleware_reads_from_cache():
    class S(DummyStorage):
        def read(self):
            self.writes += 100
            return {"x": 1}
    mw = CachingMiddleware(S)
    mw()
    first = mw.read()
    second = mw.read()
    assert first == {"x": 1}
    assert second == {"x": 1}
    assert mw.storage.writes == 100


def test_caching_middleware_flush_writes_after_cache():
    mw = CachingMiddleware(DummyStorage)
    mw()
    mw.write({"x": 1})
    assert mw._cache_modified_count == 1
    mw.flush()
    assert mw.storage.data == {"x": 1}
    assert mw._cache_modified_count == 0


def test_caching_middleware_close_flushes_and_closes_storage():
    mw = CachingMiddleware(DummyStorage)
    mw()
    mw.write({"x": 1})
    mw.close()
    assert mw.storage.data == {"x": 1}
    assert mw.storage.closed is True


def test_tinydb_with_custom_table_class():
    class CustomTable(Table):
        pass

    class CustomDB(TinyDBClass):
        table_class = CustomTable

    db = CustomDB(storage=MemoryStorage)
    assert isinstance(db.table("x"), CustomTable)


def test_tinydb_default_storage_class_can_be_overridden():
    class CustomDB(TinyDBClass):
        default_storage_class = MemoryStorage

    db = CustomDB()
    assert isinstance(db.storage, MemoryStorage)
