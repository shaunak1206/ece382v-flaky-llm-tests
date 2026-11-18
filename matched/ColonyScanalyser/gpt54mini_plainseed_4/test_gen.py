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
    original = Unique.id_count
    Unique.id_count = 0
    yield
    Unique.id_count = original


def test_identified_accepts_positive_integer_id():
    obj = Identified(3)
    assert obj.id == 3


@pytest.mark.parametrize("bad_id", [0, -1, 1.5, "1", None, [], {}])
def test_identified_rejects_invalid_ids(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_identified_id_exists_finds_matching_id_in_collection():
    items = [DummyIdentified(1), DummyIdentified(2), DummyIdentified(5)]
    assert Identified._id_exists(items, 2) is True
    assert Identified._id_exists(items, 4) is False


def test_identifiedcollection_initializes_from_unsorted_items_and_sorts_on_access():
    items = [DummyIdentified(5), DummyIdentified(1), DummyIdentified(3)]
    collection = IdentifiedCollection(items)

    assert collection.count == 3
    assert [item.id for item in collection.items] == [1, 3, 5]


def test_identifiedcollection_items_property_returns_copy_not_backing_list():
    collection = IdentifiedCollection([DummyIdentified(2), DummyIdentified(1)])
    items_copy = collection.items

    items_copy.append(DummyIdentified(3))

    assert collection.count == 2
    assert [item.id for item in collection.items] == [1, 2]


def test_identifiedcollection_items_setter_accepts_dict_values():
    collection = IdentifiedCollection({10: DummyIdentified(10), 2: DummyIdentified(2)})
    assert [item.id for item in collection.items] == [2, 10]


def test_identifiedcollection_add_appends_new_item_and_returns_it():
    collection = IdentifiedCollection()
    item = collection.add(4)

    assert isinstance(item, Identified)
    assert item.id == 4
    assert collection.count == 1
    assert collection.get_item(4) is item


def test_identifiedcollection_append_rejects_duplicate_id():
    collection = IdentifiedCollection([DummyIdentified(1)])

    with pytest.raises(ValueError, match="already exists"):
        collection.append(DummyIdentified(1))


def test_identifiedcollection_exists_and_id_exists_work_for_present_and_missing_items():
    item = DummyIdentified(7)
    collection = IdentifiedCollection([item])

    assert collection.exists(item) is True
    assert collection.id_exists(7) is True
    assert collection.id_exists(8) is False


def test_identifiedcollection_get_item_returns_item_or_none():
    collection = IdentifiedCollection([DummyIdentified(2), DummyIdentified(9)])

    found = collection.get_item(9)
    missing = collection.get_item(1)

    assert found is not None and found.id == 9
    assert missing is None


def test_identifiedcollection_remove_deletes_matching_item():
    collection = IdentifiedCollection([DummyIdentified(1), DummyIdentified(2), DummyIdentified(3)])

    collection.remove(2)

    assert [item.id for item in collection.items] == [1, 3]
    assert collection.count == 2


def test_identifiedcollection_remove_raises_keyerror_when_missing():
    collection = IdentifiedCollection([DummyIdentified(1)])

    with pytest.raises(KeyError, match="No item with ID #2"):
        collection.remove(2)


def test_named_converts_non_string_name_to_string():
    obj = Named(123)

    assert obj.name == "123"
    obj.name = None
    assert obj.name == "None"


def test_unique_auto_increments_id_and_is_read_only():
    first = Unique()
    second = Unique()

    assert first.id == 1
    assert second.id == 2
    assert Unique.id_count == 2

    first.id = 99
    assert first.id == 1


def test_timestamped_defaults_to_now_and_accepts_custom_timestamp():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()

    assert before <= obj.timestamp <= after

    custom = datetime(2020, 1, 2, 3, 4, 5)
    obj2 = TimeStamped(custom)
    assert obj2.timestamp == custom


def test_timestamp_elapsed_uses_timestamp_as_initial_when_initial_not_provided():
    ts = datetime(2021, 1, 1, 12, 0, 10)
    obj = TimeStampElapsed(timestamp=ts)

    assert obj.timestamp == ts
    assert obj.timestamp_initial == ts
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0
    assert obj.timestamp_elapsed_minutes == 0
    assert obj.timestamp_elapsed_hours == 0


def test_timestamp_elapsed_computes_differences_from_explicit_initial():
    initial = datetime(2021, 1, 1, 12, 0, 0)
    current = datetime(2021, 1, 1, 13, 30, 45)
    obj = TimeStampElapsed(timestamp=current, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=1, minutes=30, seconds=45)
    assert obj.timestamp_elapsed_seconds == 5445
    assert obj.timestamp_elapsed_minutes == 90
    assert obj.timestamp_elapsed_hours == pytest.approx(1.5125)
