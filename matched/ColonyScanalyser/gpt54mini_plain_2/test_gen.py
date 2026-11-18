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


class Item(Identified):
    pass


def test_identified_accepts_positive_integer_id():
    obj = Identified(3)
    assert obj.id == 3


def test_identified_rejects_zero_negative_and_non_integer_ids():
    for bad in [0, -1, 1.5, "3", None]:
        with pytest.raises(ValueError):
            Identified(bad)


def test_identified_id_exists_detects_matching_id_in_collection():
    items = [Item(1), Item(2), Item(5)]
    assert Identified._id_exists(items, 2) is True
    assert Identified._id_exists(items, 4) is False


def test_named_converts_any_value_to_string():
    obj = Named(123)
    assert obj.name == "123"
    obj.name = None
    assert obj.name == "None"


def test_unique_auto_increments_id_and_is_read_only():
    start = Unique.id_count
    a = Unique()
    b = Unique()

    assert a.id == start + 1
    assert b.id == start + 2

    a.id = 999
    assert a.id == start + 1


def test_unique_id_increment_advances_class_counter():
    start = Unique.id_count
    u = Unique()
    next_id = u.id_increment()

    assert next_id == start + 2
    assert Unique.id_count == start + 2


def test_timestamped_defaults_to_now_when_no_timestamp_given():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()

    assert before <= obj.timestamp <= after


def test_timestamped_uses_provided_timestamp():
    ts = datetime(2022, 1, 2, 3, 4, 5)
    obj = TimeStamped(ts)

    assert obj.timestamp == ts


def test_timestamp_elapsed_uses_same_timestamp_as_initial_when_not_provided():
    ts = datetime(2022, 1, 2, 3, 4, 5)
    obj = TimeStampElapsed(timestamp=ts)

    assert obj.timestamp == ts
    assert obj.timestamp_initial == ts
    assert obj.timestamp_elapsed == timedelta(0)


def test_timestamp_elapsed_properties_compute_expected_values():
    initial = datetime(2022, 1, 1, 0, 0, 0)
    current = initial + timedelta(hours=2, minutes=30, seconds=15)
    obj = TimeStampElapsed(timestamp=current, timestamp_initial=initial)

    assert obj.timestamp_elapsed == timedelta(hours=2, minutes=30, seconds=15)
    assert obj.timestamp_elapsed_seconds == 9015
    assert obj.timestamp_elapsed_minutes == 150
    assert obj.timestamp_elapsed_hours == pytest.approx(2.5041666667)


def test_identified_collection_initializes_from_none_and_reports_count():
    col = IdentifiedCollection()

    assert col.count == 0
    assert col.items == []


def test_identified_collection_initializes_from_list_and_returns_sorted_copy():
    col = IdentifiedCollection([Item(3), Item(1), Item(2)])

    assert [item.id for item in col.items] == [1, 2, 3]

    returned = col.items
    returned.append(Item(99))
    assert [item.id for item in col.items] == [1, 2, 3]


def test_identified_collection_initializes_from_dict_values():
    col = IdentifiedCollection({"a": Item(10), "b": Item(4)})

    assert [item.id for item in col.items] == [4, 10]


def test_identified_collection_append_and_exists_and_get_item_work():
    col = IdentifiedCollection()
    item = Item(7)

    col.append(item)

    assert col.exists(item) is True
    assert col.id_exists(7) is True
    assert col.get_item(7) is item
    assert col.get_item(8) is None


def test_identified_collection_append_duplicate_id_raises_value_error():
    col = IdentifiedCollection([Item(1)])
    with pytest.raises(ValueError, match=r"An item with ID #1 already exists"):
        col.append(Item(1))


def test_identified_collection_add_creates_item_and_appends_it():
    col = IdentifiedCollection()

    item = col.add(11)

    assert isinstance(item, Identified)
    assert item.id == 11
    assert col.count == 1
    assert col.get_item(11) is item


def test_identified_collection_remove_deletes_existing_item():
    first = Item(1)
    second = Item(2)
    col = IdentifiedCollection([first, second])

    col.remove(1)

    assert col.count == 1
    assert col.get_item(1) is None
    assert col.get_item(2) is second


def test_identified_collection_remove_missing_item_raises_key_error():
    col = IdentifiedCollection([Item(1)])

    with pytest.raises(KeyError, match=r"No item with ID #2 could be found"):
        col.remove(2)
