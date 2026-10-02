"""
Unit tests for tinydb.operations: delete, add, subtract, set, increment,
decrement, used together with Table.update.
"""
import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.operations import delete, add, subtract, set as op_set, increment, decrement


@pytest.fixture
def db():
    instance = TinyDB(storage=MemoryStorage)
    yield instance
    instance.close()


def test_delete_operation_removes_field(db):
    doc_id = db.insert({"name": "Alice", "temp": "remove-me"})
    db.update(delete("temp"), doc_ids=[doc_id])
    doc = db.get(doc_id=doc_id)
    assert "temp" not in doc
    assert doc["name"] == "Alice"


def test_add_operation_increases_numeric_field(db):
    doc_id = db.insert({"count": 5})
    db.update(add("count", 3), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["count"] == 8


def test_add_operation_with_float(db):
    doc_id = db.insert({"amount": 1.5})
    db.update(add("amount", 2.25), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["amount"] == pytest.approx(3.75)


def test_subtract_operation_decreases_numeric_field(db):
    doc_id = db.insert({"count": 10})
    db.update(subtract("count", 4), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["count"] == 6


def test_set_operation_overwrites_field(db):
    doc_id = db.insert({"status": "pending"})
    db.update(op_set("status", "done"), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["status"] == "done"


def test_set_operation_creates_new_field(db):
    doc_id = db.insert({"a": 1})
    db.update(op_set("b", 42), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["b"] == 42


def test_increment_operation(db):
    doc_id = db.insert({"visits": 0})
    db.update(increment("visits"), doc_ids=[doc_id])
    db.update(increment("visits"), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["visits"] == 2


def test_decrement_operation(db):
    doc_id = db.insert({"lives": 3})
    db.update(decrement("lives"), doc_ids=[doc_id])
    assert db.get(doc_id=doc_id)["lives"] == 2


def test_operations_applied_to_multiple_matching_documents(db):
    db.insert({"type": "item", "count": 1})
    db.insert({"type": "item", "count": 1})
    db.insert({"type": "other", "count": 1})

    updated = db.update(increment("count"), where("type") == "item")

    assert len(updated) == 2
    counts = [doc["count"] for doc in db.search(where("type") == "item")]
    assert counts == [2, 2]
    assert db.get(where("type") == "other")["count"] == 1


def test_delete_missing_field_raises_key_error(db):
    doc_id = db.insert({"a": 1})
    with pytest.raises(KeyError):
        db.update(delete("missing"), doc_ids=[doc_id])
