import pytest
from datetime import datetime, timedelta

from colonyscanalyser.base import (
    Identified,
    IdentifiedCollection,
    Named,
    Unique,
    TimeStamped,
    TimeStampElapsed,
)


class DummyIdentified(Identified):
    pass


def test_identified_accepts_positive_integer_id():
    obj = Identified(3)
    assert obj.id == 3


def test_identified_rejects_non_positive_or_non_integer_ids():
    with pytest.raises(ValueError):
        Identified(0)
    with pytest.raises(ValueError):
        Identified(-1)
    with pytest.raises(ValueError):
        Identified(1.5)
    with pytest.raises(ValueError):
        Identified("3")


def test_identified_id_exists_finds_matching_id_in_collection():
    items = [Identified(1), Identified(4), Identified(7)]
    assert Identified._id_exists(items, 4) is True
    assert Identified._id_exists(items, 2) is False


def test_identified_collection_initializes_from_list_and_sorts_items():
    items = [Identified(5), Identified(1), Identified(3)]
    collection = IdentifiedCollection(items)
    assert collection.count == 3
    assert [item.id for item in collection.items] == [1, 3, 5]


def test_identified_collection_initializes_from_dict_values():
    items = {"a": Identified(2), "b": Identified(1)}
    collection = IdentifiedCollection(items)
    assert collection.count == 2
    assert [item.id for item in collection.items] == [1, 2]


def test_identified_collection_items_property_returns_copy_not_live_view():
    items = [Identified(2), Identified(1)]
    collection = IdentifiedCollection(items)

    returned_items = collection.items
    returned_items.append(Identified(9))

    assert collection.count == 2
    assert [item.id for item in collection.items] == [1, 2]


def test_identified_collection_rejects_string_items():
    with pytest.raises(ValueError):
        IdentifiedCollection("not a collection")


def test_identified_collection_append_and_exists_and_get_item_work():
    collection = IdentifiedCollection()
    item = Identified(10)

    collection.append(item)

    assert collection.exists(item) is True
    assert collection.id_exists(10) is True
    assert collection.get_item(10) is item
    assert collection.get_item(11) is None


def test_identified_collection_append_duplicate_id_raises_value_error():
    collection = IdentifiedCollection([Identified(1)])
    with pytest.raises(ValueError):
        collection.append(Identified(1))


def test_identified_collection_add_creates_identified_item_and_appends_it():
    collection = IdentifiedCollection()
    item = collection.add(5)

    assert isinstance(item, Identified)
    assert item.id == 5
    assert collection.count == 1
    assert collection.get_item(5) is item


def test_identified_collection_remove_deletes_matching_item():
    first = Identified(1)
    second = Identified(2)
    collection = IdentifiedCollection([first, second])

    collection.remove(1)

    assert collection.count == 1
    assert collection.get_item(1) is None
    assert collection.get_item(2) is second


def test_identified_collection_remove_missing_id_raises_key_error():
    collection = IdentifiedCollection([Identified(1)])
    with pytest.raises(KeyError):
        collection.remove(99)


def test_named_converts_name_to_string():
    obj = Named(123)
    assert obj.name == "123"

    obj.name = None
    assert obj.name == "None"


def test_unique_auto_increments_id_and_is_read_only():
    start = Unique.id_count
    try:
        first = Unique()
        second = Unique()

        assert first.id == start + 1
        assert second.id == start + 2

        first.id = 999
        assert first.id == start + 1
    finally:
        Unique.id_count = start


def test_timestamped_defaults_to_now_and_accepts_explicit_timestamp():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()

    assert before <= obj.timestamp <= after

    dt = datetime(2020, 1, 2, 3, 4, 5)
    obj2 = TimeStamped(dt)
    assert obj2.timestamp == dt


def test_timestamp_elapsed_properties_compute_differences_correctly():
    initial = datetime(2020, 1, 1, 12, 0, 0)
    current = datetime(2020, 1, 1, 13, 30, 45)
    obj = TimeStampElapsed(timestamp=current, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=1, minutes=30, seconds=45)
    assert obj.timestamp_elapsed_seconds == 5445
    assert obj.timestamp_elapsed_minutes == 90
    assert obj.timestamp_elapsed_hours == pytest.approx(1.5125)


def test_timestamp_elapsed_defaults_initial_timestamp_to_timestamp():
    current = datetime(2022, 5, 1, 10, 0, 0)
    obj = TimeStampElapsed(timestamp=current)

    assert obj.timestamp == current
    assert obj.timestamp_initial == current
    assert obj.timestamp_elapsed == timedelta(0)
