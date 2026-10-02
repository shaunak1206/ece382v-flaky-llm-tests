"""Tests for storages, middlewares and update operations."""
import json
import os

import pytest

from tinydb import TinyDB, where
from tinydb import operations
from tinydb.middlewares import CachingMiddleware, Middleware
from tinydb.storages import JSONStorage, MemoryStorage, Storage, touch


def test_memory_storage_round_trip():
    storage = MemoryStorage()
    assert storage.read() is None
    data = {"_default": {"1": {"a": 1}}}
    storage.write(data)
    assert storage.read() == data
    storage.close()

    db = TinyDB(storage=MemoryStorage)
    assert db.storage.read() is None
    db.insert({"a": 1})
    assert db.storage.read() == {"_default": {"1": {"a": 1}}}
    db.close()


def test_storage_base_class_is_abstract_and_extensible():
    with pytest.raises(TypeError):
        Storage()

    class HalfStorage(Storage):
        def read(self):
            return None

    with pytest.raises(TypeError):
        HalfStorage()

    class ListStorage(Storage):
        def __init__(self):
            self.data = None
            self.closed = False

        def read(self):
            return self.data

        def write(self, data):
            self.data = data

        def close(self):
            self.closed = True

    with TinyDB(storage=ListStorage) as db:
        db.insert({"a": 1})
        storage = db.storage
        assert storage.data == {"_default": {"1": {"a": 1}}}
    assert storage.closed


def test_json_storage_used_directly(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "s.json")
    touch(path, create_dirs=True)
    assert os.path.exists(path)

    storage = JSONStorage(path)
    assert storage.read() is None
    data = {"t": {"1": {"a": 1}}}
    storage.write(data)
    assert storage.read() == data
    storage.write({"t": {}})
    assert storage.read() == {"t": {}}
    storage.close()

    broken = tmp_path / "broken.json"
    broken.write_text("this is not json")
    with TinyDB(str(broken)) as db:
        with pytest.raises(ValueError):
            db.all()


def test_caching_middleware_defers_disk_writes_until_close(tmp_path):
    path = str(tmp_path / "cached.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    db.insert({"a": 1})
    db.insert({"a": 2})
    assert os.path.getsize(path) == 0
    assert len(db) == 2
    assert db.get(doc_id=2) == {"a": 2}
    db.close()
    with open(path) as fh:
        assert json.load(fh) == {"_default": {"1": {"a": 1}, "2": {"a": 2}}}
    with TinyDB(path) as reopened:
        assert len(reopened) == 2


def test_caching_middleware_flushes_after_write_cache_size(tmp_path, monkeypatch):
    monkeypatch.setattr(CachingMiddleware, "WRITE_CACHE_SIZE", 3)
    path = str(tmp_path / "cached.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    db.insert({"a": 1})
    db.insert({"a": 2})
    assert os.path.getsize(path) == 0
    db.insert({"a": 3})
    assert os.path.getsize(path) > 0
    with open(path) as fh:
        assert len(json.load(fh)["_default"]) == 3
    db.close()


def test_caching_middleware_manual_flush_and_attribute_forwarding():
    middleware = CachingMiddleware(MemoryStorage)
    db = TinyDB(storage=middleware)
    assert db.storage is middleware
    assert middleware.memory is None
    middleware.flush()
    assert middleware.storage.memory is None
    db.insert({"a": 1})
    assert middleware.storage.memory is None
    middleware.flush()
    assert middleware.storage.memory == {"_default": {"1": {"a": 1}}}

    plain = TinyDB(storage=Middleware(MemoryStorage))
    plain.insert({"b": 2})
    assert plain.all() == [{"b": 2}]
    db.close()
    plain.close()


def test_nested_middlewares_flush_down_to_the_storage():
    inner = CachingMiddleware(MemoryStorage)
    outer = CachingMiddleware(inner)
    db = TinyDB(storage=outer)
    db.insert({"a": 1})
    assert inner.storage.memory is None
    db.close()
    assert inner.storage.memory == {"_default": {"1": {"a": 1}}}


@pytest.mark.parametrize(
    "op,initial,expected",
    [
        (operations.delete("a"), {"a": 1, "b": 2}, {"b": 2}),
        (operations.add("a", 5), {"a": 1}, {"a": 6}),
        (operations.add("a", 0.5), {"a": 1}, {"a": 1.5}),
        (operations.subtract("a", 5), {"a": 10}, {"a": 5}),
        (operations.set("a", "x"), {"a": 1}, {"a": "x"}),
        (operations.set("new", 1), {"a": 1}, {"a": 1, "new": 1}),
        (operations.increment("a"), {"a": 1}, {"a": 2}),
        (operations.decrement("a"), {"a": 1}, {"a": 0}),
    ],
)
def test_operations_mutate_documents_in_place(op, initial, expected):
    op(initial)
    assert initial == expected


def test_operations_applied_through_the_database():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple(
        [
            {"name": "a", "n": 1, "junk": True},
            {"name": "b", "n": 5, "junk": True},
        ]
    )
    db.update(operations.increment("n"), where("name") == "a")
    db.update(operations.add("n", 10), where("name") == "b")
    db.update(operations.delete("junk"))
    assert db.get(where("name") == "a") == {"name": "a", "n": 2}
    assert db.get(where("name") == "b") == {"name": "b", "n": 15}

    db.update(operations.decrement("n"), where("name") == "a")
    db.update(operations.subtract("n", 5), where("name") == "b")
    db.update(operations.set("name", "z"), where("n") == 1)
    assert sorted(d["n"] for d in db.all()) == [1, 10]
    assert db.contains(where("name") == "z")

    with pytest.raises(KeyError):
        operations.delete("missing")({})
    with pytest.raises(KeyError):
        operations.increment("missing")({})
    db.close()
