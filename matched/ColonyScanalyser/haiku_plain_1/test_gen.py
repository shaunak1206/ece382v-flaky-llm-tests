import datetime as dt
from datetime import datetime, timedelta

import pytest

from colonyscanalyser.base import (
    Identified,
    IdentifiedCollection,
    Named,
    TimeStamped,
    TimeStampElapsed,
    Unique,
)


# ---------------------------------------------------------------------------
# Identified
# ---------------------------------------------------------------------------

def test_identified_stores_valid_id():
    obj = Identified(id=7)
    assert obj.id == 7


def test_identified_rejects_non_positive_and_non_int_ids():
    for bad in (0, -1, -100, "3", 2.0, None):
        with pytest.raises(ValueError):
            Identified(id=bad)


def test_identified_id_setter_updates_value_and_validates():
    obj = Identified(id=1)
    obj.id = 42
    assert obj.id == 42
    with pytest.raises(ValueError):
        obj.id = 0
    assert obj.id == 42  # unchanged after failed assignment


def test_identified_static_id_is_valid():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid(-5) is False
    assert Identified._id_is_valid("1") is False


def test_identified_static_id_exists():
    items = [Identified(id=1), Identified(id=3)]
    assert Identified._id_exists(items, 3) is True
    assert Identified._id_exists(items, 2) is False
    assert Identified._id_exists([], 1) is False


# ---------------------------------------------------------------------------
# Unique
# ---------------------------------------------------------------------------

def test_unique_assigns_increasing_ids_and_is_identified():
    first = Unique()
    second = Unique()
    assert isinstance(first, Identified)
    assert first.id > 0
    assert second.id > first.id
    assert second.id == first.id + 1


def test_unique_id_is_read_only():
    obj = Unique()
    original = obj.id
    obj.id = original + 1000
    assert obj.id == original


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------

def test_named_stores_name():
    obj = Named(name="Plate A")
    assert obj.name == "Plate A"


def test_named_converts_non_string_to_string():
    obj = Named(name=123)
    assert obj.name == "123"
    assert isinstance(obj.name, str)


# ---------------------------------------------------------------------------
# TimeStamped
# ---------------------------------------------------------------------------

def test_timestamped_uses_given_timestamp():
    ts = datetime(2024, 5, 1, 12, 0, 0)
    obj = TimeStamped(timestamp=ts)
    assert obj.timestamp == ts


def test_timestamped_default_is_current_time():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()
    assert before <= obj.timestamp <= after


def test_timestamped_setter_updates_timestamp():
    obj = TimeStamped(timestamp=datetime(2024, 1, 1))
    new_ts = datetime(2025, 6, 15, 8, 30)
    obj.timestamp = new_ts
    assert obj.timestamp == new_ts


# ---------------------------------------------------------------------------
# TimeStampElapsed
# ---------------------------------------------------------------------------

def test_timestamp_elapsed_defaults_initial_to_timestamp():
    ts = datetime(2024, 3, 10, 9, 0, 0)
    obj = TimeStampElapsed(timestamp=ts)
    assert obj.timestamp_initial == ts
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0


def test_timestamp_elapsed_computes_units():
    initial = datetime(2024, 1, 1, 0, 0, 0)
    current = initial + timedelta(hours=2, minutes=30)
    obj = TimeStampElapsed(timestamp=current, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=2, minutes=30)
    assert obj.timestamp_elapsed_seconds == pytest.approx(9000.0)
    assert obj.timestamp_elapsed_minutes == 150
    assert obj.timestamp_elapsed_hours == pytest.approx(2.5)


def test_timestamp_elapsed_updates_when_timestamp_changes():
    initial = datetime(2024, 1, 1, 0, 0, 0)
    obj = TimeStampElapsed(timestamp=initial, timestamp_initial=initial)
    assert obj.timestamp_elapsed_minutes == 0

    obj.timestamp = initial + timedelta(minutes=45)
    assert obj.timestamp_elapsed_minutes == 45
    assert obj.timestamp_elapsed_seconds == pytest.approx(2700.0)


# ---------------------------------------------------------------------------
# IdentifiedCollection
# ---------------------------------------------------------------------------

def test_collection_defaults_to_empty():
    collection = IdentifiedCollection()
    assert collection.count == 0
    assert collection.items == []


def test_collection_items_are_sorted_by_id():
    collection = IdentifiedCollection([Identified(id=3), Identified(id=1), Identified(id=2)])
    assert [item.id for item in collection.items] == [1, 2, 3]
    assert collection.count == 3


def test_collection_init_from_dict_uses_values():
    collection = IdentifiedCollection({"b": Identified(id=2), "a": Identified(id=1)})
    assert [item.id for item in collection.items] == [1, 2]


def test_collection_init_copies_source_list():
    source = [Identified(id=1), Identified(id=2)]
    collection = IdentifiedCollection(source)
    source.append(Identified(id=3))
    assert collection.count == 2
    assert not collection.id_exists(3)


def test_collection_rejects_string_items():
    with pytest.raises(ValueError):
        IdentifiedCollection("not a collection")


def test_collection_add_creates_and_stores_item():
    collection = IdentifiedCollection()
    item = collection.add(5)
    assert isinstance(item, Identified)
    assert item.id == 5
    assert collection.count == 1
    assert collection.exists(item)


def test_collection_add_rejects_invalid_id():
    collection = IdentifiedCollection()
    with pytest.raises(ValueError):
        collection.add(0)
    assert collection.count == 0


def test_collection_append_duplicate_raises_and_keeps_count():
    collection = IdentifiedCollection([Identified(id=1)])
    with pytest.raises(ValueError):
        collection.append(Identified(id=1))
    assert collection.count == 1


def test_collection_id_exists_and_get_item():
    target = Identified(id=4)
    collection = IdentifiedCollection([Identified(id=2), target])

    assert collection.id_exists(4) is True
    assert collection.id_exists(9) is False
    assert collection.get_item(4) is target
    assert collection.get_item(9) is None


def test_collection_remove_existing_and_missing():
    collection = IdentifiedCollection([Identified(id=1), Identified(id=2)])

    collection.remove(1)
    assert collection.count == 1
    assert not collection.id_exists(1)
    assert collection.id_exists(2)

    with pytest.raises(KeyError):
        collection.remove(1)
    assert collection.count == 1
