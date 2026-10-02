"""Tests for TinyDB backed by real JSON files, plus database level behaviour."""
import json
import os

import pytest

from tinydb import TinyDB, where
from tinydb.storages import JSONStorage, MemoryStorage


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.json")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_opening_creates_the_file(db_path):
    assert not os.path.exists(db_path)
    db = TinyDB(db_path)
    try:
        assert os.path.exists(db_path)
    finally:
        db.close()


def test_empty_file_is_an_empty_database(db_path):
    with TinyDB(db_path) as db:
        assert db.all() == []
        assert len(db) == 0
        assert db.tables() == set()


def test_insert_is_written_to_disk(db_path):
    with TinyDB(db_path) as db:
        doc_id = db.insert({"name": "alice", "age": 30})
    assert doc_id == 1
    assert read_json(db_path) == {"_default": {"1": {"name": "alice", "age": 30}}}


def test_data_survives_reopening(db_path):
    with TinyDB(db_path) as db:
        db.insert_multiple([{"n": i} for i in range(5)])
    with TinyDB(db_path) as db:
        docs = db.all()
        assert len(db) == 5
        assert [d["n"] for d in docs] == [0, 1, 2, 3, 4]
        assert [d.doc_id for d in docs] == [1, 2, 3, 4, 5]
        assert all(isinstance(d.doc_id, int) for d in docs)


def test_next_id_continues_after_reopening(db_path):
    with TinyDB(db_path) as db:
        db.insert_multiple([{"a": 1}, {"a": 2}])
    with TinyDB(db_path) as db:
        assert db.insert({"a": 3}) == 3


def test_loads_a_handwritten_file(db_path):
    with open(db_path, "w") as fh:
        json.dump(
            {"_default": {"1": {"a": 1}, "2": {"a": 2}}, "other": {"7": {"b": True}}},
            fh,
        )
    with TinyDB(db_path) as db:
        assert db.tables() == {"_default", "other"}
        assert db.get(doc_id=2) == {"a": 2}
        assert db.table("other").get(doc_id=7) == {"b": True}
        assert db.table("other").insert({"b": False}) == 8


def test_context_manager_closes_the_file_handle(db_path):
    with TinyDB(db_path) as db:
        db.insert({"a": 1})
        handle = db.storage._handle
        assert not handle.closed
    assert handle.closed


def test_create_dirs_builds_missing_parents(tmp_path):
    path = str(tmp_path / "a" / "b" / "db.json")
    with TinyDB(path, create_dirs=True) as db:
        db.insert({"x": 1})
    assert read_json(path)["_default"]["1"] == {"x": 1}


def test_missing_parent_directory_without_create_dirs(tmp_path):
    path = str(tmp_path / "missing" / "db.json")
    with pytest.raises(FileNotFoundError):
        TinyDB(path)


def test_read_only_mode_allows_reads_but_not_writes(db_path):
    with open(db_path, "w") as fh:
        json.dump({"_default": {"1": {"a": 1}}}, fh)
    with TinyDB(db_path, access_mode="r") as db:
        assert db.all() == [{"a": 1}]
        with pytest.raises(IOError):
            db.insert({"b": 2})


def test_unusual_access_mode_emits_a_warning(db_path):
    with pytest.warns(UserWarning):
        storage = JSONStorage(db_path, access_mode="a+")
    storage.close()


def test_json_kwargs_are_forwarded_to_dumps(db_path):
    with TinyDB(db_path, indent=2) as db:
        db.insert({"a": 1})
    with open(db_path) as fh:
        text = fh.read()
    assert "\n" in text
    assert '  "_default"' in text


def test_unicode_round_trip_with_encoding(db_path):
    with TinyDB(db_path, encoding="utf-8", ensure_ascii=False) as db:
        db.insert({"name": "José", "city": "München"})
    with open(db_path, "rb") as fh:
        raw = fh.read().decode("utf-8")
    assert "José" in raw
    with TinyDB(db_path, encoding="utf-8") as db:
        assert db.all()[0]["city"] == "München"


def test_file_is_truncated_when_data_shrinks(db_path):
    with TinyDB(db_path) as db:
        db.insert({"blob": "x" * 5000})
        db.insert({"small": 1})
        size_before = os.path.getsize(db_path)
        db.truncate()
        size_after = os.path.getsize(db_path)
    assert size_after < size_before
    assert read_json(db_path) == {"_default": {}}


def test_multiple_tables_are_persisted(db_path):
    with TinyDB(db_path) as db:
        db.table("users").insert({"name": "a"})
        db.table("posts").insert_multiple([{"t": 1}, {"t": 2}])
        db.insert({"d": 1})
    with TinyDB(db_path) as db:
        assert db.tables() == {"users", "posts", "_default"}
        assert len(db.table("users")) == 1
        assert len(db.table("posts")) == 2
        assert len(db) == 1


def test_drop_table_removes_it_from_the_file(db_path):
    with TinyDB(db_path) as db:
        db.table("a").insert({"x": 1})
        db.table("b").insert({"y": 2})
        db.drop_table("a")
        assert db.tables() == {"b"}
    assert set(read_json(db_path)) == {"b"}


def test_drop_table_on_unknown_table_is_a_noop(db_path):
    with TinyDB(db_path) as db:
        db.drop_table("nothing")
        db.insert({"a": 1})
        db.drop_table("nothing")
        assert len(db) == 1


def test_drop_tables_resets_the_database(db_path):
    with TinyDB(db_path) as db:
        db.table("a").insert({"x": 1})
        db.insert({"y": 2})
        db.drop_tables()
        assert db.tables() == set()
        assert db.all() == []
    assert read_json(db_path) == {}


def test_repr_summarises_tables(db_path):
    with TinyDB(db_path) as db:
        db.insert({"a": 1})
        text = repr(db)
    assert text.startswith("<TinyDB ")
    assert "tables_count=1" in text
    assert "default_table_documents_count=1" in text


def test_custom_default_table_name():
    class MyDB(TinyDB):
        default_table_name = "main"

    with MyDB(storage=MemoryStorage) as db:
        db.insert({"a": 1})
        assert db.tables() == {"main"}
        assert len(db.table("main")) == 1


def test_persist_empty_table_is_stored_immediately():
    with TinyDB(storage=MemoryStorage) as db:
        db.table("empty", persist_empty=True)
        db.table("lazy")
        assert "empty" in db.tables()
        assert "lazy" not in db.tables()
