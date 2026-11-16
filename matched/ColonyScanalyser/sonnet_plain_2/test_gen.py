from datetime import datetime, timedelta

import pytest

from colonyscanalyser.base import (
    Identified,
    IdentifiedCollection,
    Named,
    Unique,
    TimeStamped,
    TimeStampElapsed,
)


# ---------------------------------------------------------------- Identified

def test_identified_stores_valid_id():
    item = Identified(5)
    assert item.id == 5


@pytest.mark.parametrize("bad_id", [0, -1, -100, 1.5, "1", None, [1]])
def test_identified_rejects_invalid_id_on_init(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_identified_id_setter_validates_and_keeps_old_value():
    item = Identified(3)
    item.id = 10
    assert item.id == 10
    with pytest.raises(ValueError):
        item.id = 0
    assert item.id == 10


def test_identified_static_helpers():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid("2") is False

    collection = [Identified(1), Identified(4)]
    assert Identified._id_exists(collection, 4) is True
    assert Identified._id_exists(collection, 2) is False
    assert Identified._id_exists([], 1) is False


# ---------------------------------------------------- IdentifiedCollection

def test_collection_defaults_to_empty():
    collection = IdentifiedCollection()
    assert collection.count == 0
    assert collection.items == []


def test_collection_init_copies_and_sorts_items():
    a, b, c = Identified(3), Identified(1), Identified(2)
    source = [a, b, c]
    collection = IdentifiedCollection(source)

    assert collection.items == [b, c, a]
    assert collection.count == 3

    # Changing the source list must not affect the collection
    source.append(Identified(9))
    assert collection.count == 3


def test_collection_accepts_dict_values():
    a, b = Identified(2), Identified(1)
    collection = IdentifiedCollection({"x": a, "y": b})
    assert collection.items == [b, a]


@pytest.mark.parametrize("bad", ["abc", 5, 3.2])
def test_collection_rejects_non_collections(bad):
    with pytest.raises(ValueError):
        IdentifiedCollection(bad)


def test_collection_items_returns_copy():
    collection = IdentifiedCollection([Identified(1)])
    items = collection.items
    items.append(Identified(2))
    assert collection.count == 1


def test_collection_add_creates_and_stores_item():
    collection = IdentifiedCollection()
    item = collection.add(7)

    assert isinstance(item, Identified)
    assert item.id == 7
    assert collection.count == 1
    assert collection.get_item(7) is item


def test_collection_add_duplicate_id_raises():
    collection = IdentifiedCollection()
    collection.add(1)
    with pytest.raises(ValueError):
        collection.add(1)
    assert collection.count == 1


def test_collection_append_and_exists():
    collection = IdentifiedCollection()
    item = Identified(4)
    assert collection.exists(item) is False

    collection.append(item)
    assert collection.exists(item) is True
    assert collection.id_exists(4) is True
    assert collection.id_exists(5) is False

    with pytest.raises(ValueError):
        collection.append(Identified(4))


def test_collection_get_item_missing_returns_none():
    collection = IdentifiedCollection([Identified(1)])
    assert collection.get_item(2) is None


def test_collection_remove_existing_and_missing():
    a, b = Identified(1), Identified(2)
    collection = IdentifiedCollection([a, b])

    collection.remove(1)
    assert collection.items == [b]
    assert collection.id_exists(1) is False

    with pytest.raises(KeyError):
        collection.remove(1)
    with pytest.raises(KeyError):
        collection.remove(99)


# --------------------------------------------------------------------- Named

def test_named_stores_name_and_coerces_to_string():
    assert Named("colony").name == "colony"
    assert Named(42).name == "42"

    named = Named("a")
    named.name = 3.5
    assert named.name == "3.5"


# -------------------------------------------------------------------- Unique

def test_unique_ids_auto_increment():
    first = Unique()
    second = Unique()
    third = Unique()

    assert isinstance(first, Identified)
    assert second.id == first.id + 1
    assert third.id == second.id + 1
    assert Unique.id_count == third.id


def test_unique_id_is_read_only():
    item = Unique()
    original = item.id
    item.id = original + 100
    assert item.id == original


def test_unique_works_in_collection():
    collection = IdentifiedCollection()
    a, b = Unique(), Unique()
    collection.append(a)
    collection.append(b)
    assert collection.items == [a, b]
    with pytest.raises(ValueError):
        collection.append(a)


# --------------------------------------------------------------- TimeStamped

def test_timestamped_default_and_explicit_timestamp():
    before = datetime.now()
    stamped = TimeStamped()
    after = datetime.now()
    assert before <= stamped.timestamp <= after

    explicit = datetime(2020, 1, 2, 3, 4, 5)
    stamped = TimeStamped(explicit)
    assert stamped.timestamp == explicit

    new = datetime(2021, 6, 7)
    stamped.timestamp = new
    assert stamped.timestamp == new


# ----------------------------------------------------------- TimeStampElapsed

def test_elapsed_defaults_to_zero():
    obj = TimeStampElapsed()
    assert obj.timestamp_initial == obj.timestamp
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0
    assert obj.timestamp_elapsed_minutes == 0
    assert obj.timestamp_elapsed_hours == 0


def test_elapsed_initial_defaults_to_timestamp():
    ts = datetime(2020, 5, 5, 12, 0, 0)
    obj = TimeStampElapsed(timestamp=ts)
    assert obj.timestamp == ts
    assert obj.timestamp_initial == ts
    assert obj.timestamp_elapsed == timedelta(0)


def test_elapsed_calculations():
    initial = datetime(2020, 1, 1, 0, 0, 0)
    ts = initial + timedelta(hours=2, minutes=30, seconds=30)
    obj = TimeStampElapsed(timestamp=ts, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=2, minutes=30, seconds=30)
    assert obj.timestamp_elapsed_seconds == 9030
    assert obj.timestamp_elapsed_minutes == 150  # truncated to int
    assert obj.timestamp_elapsed_hours == pytest.approx(9030 / 3600)


def test_elapsed_updates_when_timestamps_change():
    initial = datetime(2020, 1, 1)
    obj = TimeStampElapsed(timestamp=initial + timedelta(hours=1), timestamp_initial=initial)
    assert obj.timestamp_elapsed_hours == pytest.approx(1.0)

    obj.timestamp = initial + timedelta(hours=3)
    assert obj.timestamp_elapsed_hours == pytest.approx(3.0)

    obj.timestamp_initial = initial + timedelta(hours=1)
    assert obj.timestamp_elapsed_hours == pytest.approx(2.0)
    assert isinstance(obj, TimeStamped)
