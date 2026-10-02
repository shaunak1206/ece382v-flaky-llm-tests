"""
Unit tests for tinydb.middlewares: CachingMiddleware wrapping both
MemoryStorage and JSONStorage (real temp files).
"""
import json

import pytest

from tinydb import TinyDB, where
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import MemoryStorage, JSONStorage


def test_caching_middleware_wraps_memory_storage():
    db = TinyDB(storage=CachingMiddleware(MemoryStorage))
    try:
        db.insert({"a": 1})
        assert db.all()[0]["a"] == 1
    finally:
        db.close()


def test_caching_middleware_read_returns_none_when_empty():
    db = TinyDB(storage=CachingMiddleware(MemoryStorage))
    try:
        assert db.all() == []
    finally:
        db.close()


def test_caching_middleware_does_not_write_immediately(tmp_path):
    path = str(tmp_path / "cached.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    try:
        db.insert({"a": 1})
        # Underlying file should still be empty (untouched) since the
        # write cache has not been flushed yet.
        with open(path) as f:
            content = f.read()
        assert content == ""
    finally:
        db.close()


def test_caching_middleware_flush_writes_to_disk(tmp_path):
    path = str(tmp_path / "flushed.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    db.insert({"a": 1})
    db.storage.flush()

    with open(path) as f:
        data = json.load(f)
    assert data["_default"]["1"]["a"] == 1
    db.close()


def test_caching_middleware_close_flushes_pending_writes(tmp_path):
    path = str(tmp_path / "closed.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    db.insert({"a": 1})
    db.close()

    with open(path) as f:
        data = json.load(f)
    assert data["_default"]["1"]["a"] == 1


def test_caching_middleware_reads_from_cache_not_storage(tmp_path):
    path = str(tmp_path / "cache_read.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    try:
        db.insert({"a": 1})
        # A second read should be served from cache and reflect the write
        # even though it has not been flushed to disk yet.
        assert len(db) == 1
        assert db.all()[0]["a"] == 1
    finally:
        db.close()


def test_caching_middleware_forwards_unknown_attributes(tmp_path):
    path = str(tmp_path / "forward.json")
    middleware = CachingMiddleware(JSONStorage)
    db = TinyDB(path, storage=middleware)
    try:
        # Middleware.__getattr__ should forward to the underlying storage;
        # JSONStorage instances expose a `_handle` attribute.
        assert hasattr(db.storage, "_handle")
    finally:
        db.close()


def test_caching_middleware_multiple_writes_accumulate_in_cache(tmp_path):
    path = str(tmp_path / "accum.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    try:
        for i in range(5):
            db.insert({"n": i})
        assert len(db) == 5
        # Nothing flushed yet
        assert db.storage._cache_modified_count == 5
        db.storage.flush()
        assert db.storage._cache_modified_count == 0
    finally:
        db.close()


def test_caching_middleware_search_and_update_work_before_flush(tmp_path):
    path = str(tmp_path / "search_before_flush.json")
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    try:
        doc_id = db.insert({"name": "Alice", "age": 30})
        db.update({"age": 31}, where("name") == "Alice")
        assert db.get(doc_id=doc_id)["age"] == 31
    finally:
        db.close()
