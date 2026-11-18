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


@pytest.fixture(autouse=True)
def reset_unique_id_count():
    Unique.id_count = 0
    yield
    Unique.id_count = 0


def test_identified_accepts_positive_integer_id():
    obj = Identified(1)
    assert obj.id == 1


@pytest.mark.parametrize("bad_id", [0, -1, -100, 1.5, "1", None, [], {}])
def test_identified_rejects_invalid_ids(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_id_is_valid_and_id_exists_helpers():
    items = [DummyIdentified(1), DummyIdentified(3), DummyIdentified(5)]
    assert Identified._id_is_valid(7) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid("7") is False
    assert Identified._id_exists(items, 3) is True
    assert Identified._id_exists(items, 2) is False


def test_identified_collection_initializes_from_list_and_sorts_items():
    items = [DummyIdentified(5), DummyIdentified(1), DummyIdentified(3)]
    collection = IdentifiedCollection(items)

    assert collection.count == 3
    assert [item.id for item in collection.items] == [1, 3, 5]


def test_identified_collection_initializes_from_dict_values():
    items = {
        "b": DummyIdentified(2),
        "a": DummyIdentified(1),
        "c": DummyIdentified(3),
    }
    collection = IdentifiedCollection(items)

    assert collection.count == 3
    assert [item.id for item in collection.items] == [1, 2, 3]


def test_identified_collection_items_returns_copy_not_live_list():
    collection = IdentifiedCollection([DummyIdentified(2), DummyIdentified(1)])
    returned_items = collection.items

    returned_items.append(DummyIdentified(99))

    assert collection.count == 2
    assert [item.id for item in collection.items] == [1, 2]


def test_identified_collection_rejects_string_items():
    with pytest.raises(ValueError):
        IdentifiedCollection("not a collection")


def test_identified_collection_add_creates_identified_and_appends():
    collection = IdentifiedCollection()
    item = collection.add(4)

    assert isinstance(item, Identified)
    assert item.id == 4
    assert collection.count == 1
    assert collection.get_item(4) is item


def test_identified_collection_append_and_exists_and_duplicate_rejected():
    collection = IdentifiedCollection([DummyIdentified(1)])
    new_item = DummyIdentified(2)

    assert collection.exists(new_item) is False
    collection.append(new_item)
    assert collection.exists(new_item) is True
    assert collection.id_exists(2) is True

    with pytest.raises(ValueError):
        collection.append(DummyIdentified(2))


def test_identified_collection_get_item_returns_none_when_missing():
    collection = IdentifiedCollection([DummyIdentified(1)])
    assert collection.get_item(999) is None


def test_identified_collection_remove_deletes_item():
    item1 = DummyIdentified(1)
    item2 = DummyIdentified(2)
    collection = IdentifiedCollection([item1, item2])

    collection.remove(1)

    assert collection.count == 1
    assert collection.get_item(1) is None
    assert collection.get_item(2) is item2


def test_identified_collection_remove_missing_raises_key_error():
    collection = IdentifiedCollection([DummyIdentified(1)])

    with pytest.raises(KeyError):
        collection.remove(2)


def test_named_converts_name_to_string():
    obj = Named(123)
    assert obj.name == "123"

    obj.name = None
    assert obj.name == "None"


def test_unique_auto_increments_and_id_is_read_only():
    a = Unique()
    b = Unique()
    c = Unique()

    assert a.id == 1
    assert b.id == 2
    assert c.id == 3

    a.id = 999
    assert a.id == 1
    assert Unique.id_count == 3


def test_timestamp_defaults_to_now_and_accepts_explicit_value():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()

    assert before <= obj.timestamp <= after

    dt = datetime(2024, 1, 2, 3, 4, 5)
    obj.timestamp = dt
    assert obj.timestamp == dt


def test_timestamp_elapsed_uses_same_timestamp_as_initial_by_default():
    dt = datetime(2024, 1, 1, 12, 0, 0)
    obj = TimeStampElapsed(timestamp=dt)

    assert obj.timestamp == dt
    assert obj.timestamp_initial == dt
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0
    assert obj.timestamp_elapsed_minutes == 0
    assert obj.timestamp_elapsed_hours == 0.0


def test_timestamp_elapsed_computes_difference_and_conversions():
    initial = datetime(2024, 1, 1, 10, 0, 0)
    current = datetime(2024, 1, 1, 12, 30, 0)
    obj = TimeStampElapsed(timestamp=current, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=2, minutes=30)
    assert obj.timestamp_elapsed_seconds == 9000
    assert obj.timestamp_elapsed_minutes == 150
    assert obj.timestamp_elapsed_hours == 2.5
