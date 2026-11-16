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
    obj = Identified(5)
    assert obj.id == 5


@pytest.mark.parametrize("bad_id", [0, -1, -100])
def test_identified_rejects_non_positive_id(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


@pytest.mark.parametrize("bad_id", ["3", 2.5, None])
def test_identified_rejects_non_integer_id(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_identified_id_setter_updates_value():
    obj = Identified(1)
    obj.id = 7
    assert obj.id == 7


def test_identified_invalid_id_setter_keeps_original_value():
    obj = Identified(4)
    with pytest.raises(ValueError):
        obj.id = -2
    assert obj.id == 4


def test_id_is_valid_static_method():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(999) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid(-3) is False
    assert Identified._id_is_valid("1") is False


def test_id_exists_static_method():
    items = [Identified(1), Identified(2)]
    assert Identified._id_exists(items, 2) is True
    assert Identified._id_exists(items, 3) is False
    assert Identified._id_exists([], 1) is False


# ---------------------------------------------------------------------------
# IdentifiedCollection
# ---------------------------------------------------------------------------

def test_collection_default_is_empty():
    coll = IdentifiedCollection()
    assert coll.count == 0
    assert coll.items == []


def test_collection_items_are_sorted_by_id():
    coll = IdentifiedCollection([Identified(3), Identified(1), Identified(2)])
    assert [item.id for item in coll.items] == [1, 2, 3]


def test_collection_accepts_dict_and_uses_values():
    coll = IdentifiedCollection({"a": Identified(3), "b": Identified(1)})
    assert coll.count == 2
    assert [item.id for item in coll.items] == [1, 3]


def test_collection_none_items_yields_empty_collection():
    coll = IdentifiedCollection(None)
    assert coll.count == 0


def test_collection_rejects_string_items():
    with pytest.raises(ValueError):
        IdentifiedCollection("not a collection")


def test_collection_copies_input_list():
    source = [Identified(2)]
    coll = IdentifiedCollection(source)
    source.append(Identified(1))
    assert coll.count == 1


def test_add_creates_and_stores_new_item():
    coll = IdentifiedCollection()
    item = coll.add(10)
    assert isinstance(item, Identified)
    assert item.id == 10
    assert coll.count == 1
    assert coll.get_item(10) is item


def test_append_duplicate_id_raises():
    coll = IdentifiedCollection([Identified(1)])
    with pytest.raises(ValueError):
        coll.append(Identified(1))
    assert coll.count == 1


def test_exists_and_id_exists():
    existing = Identified(2)
    coll = IdentifiedCollection([existing])
    assert coll.exists(Identified(2)) is True
    assert coll.exists(Identified(5)) is False
    assert coll.id_exists(2) is True
    assert coll.id_exists(5) is False


def test_get_item_returns_match_or_none():
    first = Identified(1)
    coll = IdentifiedCollection([first])
    assert coll.get_item(1) is first
    assert coll.get_item(42) is None


def test_remove_existing_item():
    coll = IdentifiedCollection([Identified(1), Identified(2)])
    coll.remove(1)
    assert coll.count == 1
    assert coll.id_exists(1) is False
    assert coll.id_exists(2) is True


def test_remove_missing_item_raises_keyerror():
    coll = IdentifiedCollection([Identified(1)])
    with pytest.raises(KeyError):
        coll.remove(99)


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------

def test_named_stores_name():
    obj = Named("colony")
    assert obj.name == "colony"


def test_named_setter_converts_value_to_str():
    obj = Named("start")
    obj.name = 123
    assert obj.name == "123"


# ---------------------------------------------------------------------------
# Unique
# ---------------------------------------------------------------------------

def test_unique_is_identified_and_ids_increment():
    first = Unique()
    second = Unique()
    assert isinstance(first, Identified)
    assert second.id == first.id + 1


def test_unique_id_increment_updates_class_counter():
    before = Unique.id_count
    obj = Unique()
    assert Unique.id_count == before + 1
    assert obj.id == Unique.id_count


def test_unique_id_is_read_only():
    obj = Unique()
    original = obj.id
    obj.id = 999
    assert obj.id == original


# ---------------------------------------------------------------------------
# TimeStamped
# ---------------------------------------------------------------------------

def test_timestamped_defaults_to_current_time():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()
    assert before <= obj.timestamp <= after


def test_timestamped_stores_explicit_timestamp():
    moment = datetime(2024, 1, 1, 12, 0, 0)
    obj = TimeStamped(moment)
    assert obj.timestamp == moment


def test_timestamped_setter_replaces_timestamp():
    obj = TimeStamped(datetime(2024, 1, 1))
    new_moment = datetime(2024, 6, 1)
    obj.timestamp = new_moment
    assert obj.timestamp == new_moment


# ---------------------------------------------------------------------------
# TimeStampElapsed
# ---------------------------------------------------------------------------

def test_timestamp_elapsed_defaults_to_zero():
    obj = TimeStampElapsed()
    assert obj.timestamp_initial == obj.timestamp
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0


def test_timestamp_elapsed_computes_difference_and_units():
    start = datetime(2024, 1, 1, 12, 0, 0)
    end = start + timedelta(seconds=150)
    obj = TimeStampElapsed(timestamp=end, timestamp_initial=start)

    assert obj.timestamp_elapsed == timedelta(seconds=150)
    assert obj.timestamp_elapsed_seconds == 150.0
    assert obj.timestamp_elapsed_minutes == 2
    assert obj.timestamp_elapsed_hours == pytest.approx(150 / 3600)


def test_timestamp_elapsed_hours_exact_value():
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(seconds=5400)
    obj = TimeStampElapsed(timestamp=end, timestamp_initial=start)
    assert obj.timestamp_elapsed_hours == 1.5


def test_timestamp_elapsed_updates_when_timestamp_changes():
    start = datetime(2024, 1, 1, 0, 0, 0)
    obj = TimeStampElapsed(timestamp=start, timestamp_initial=start)
    obj.timestamp = start + timedelta(seconds=60)
    assert obj.timestamp_elapsed_seconds == 60.0
    assert obj.timestamp_elapsed_minutes == 1


def test_timestamp_elapsed_is_timestamped_and_can_be_negative():
    start = datetime(2024, 1, 1, 0, 0, 0)
    earlier = start - timedelta(seconds=30)
    obj = TimeStampElapsed(timestamp=earlier, timestamp_initial=start)
    assert isinstance(obj, TimeStamped)
    assert obj.timestamp_elapsed_seconds == -30.0
