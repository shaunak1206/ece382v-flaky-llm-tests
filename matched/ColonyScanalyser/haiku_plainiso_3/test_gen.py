import pytest
from datetime import datetime, timedelta

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
    obj = Identified(id=5)
    assert obj.id == 5


@pytest.mark.parametrize("bad_id", [0, -1, -100])
def test_identified_rejects_zero_and_negative_ids(bad_id):
    with pytest.raises(ValueError):
        Identified(id=bad_id)


@pytest.mark.parametrize("bad_id", ["1", 1.0, None, [1]])
def test_identified_rejects_non_integer_ids(bad_id):
    with pytest.raises(ValueError):
        Identified(id=bad_id)


def test_identified_id_setter_accepts_valid_value():
    obj = Identified(id=1)
    obj.id = 42
    assert obj.id == 42


def test_identified_id_setter_rejects_invalid_value_and_keeps_old_id():
    obj = Identified(id=3)
    with pytest.raises(ValueError):
        obj.id = 0
    assert obj.id == 3


def test_identified_static_id_is_valid():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid("2") is False


def test_identified_static_id_exists():
    items = [Identified(id=1), Identified(id=7)]
    assert Identified._id_exists(items, 7) is True
    assert Identified._id_exists(items, 2) is False


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------

def test_named_stores_name():
    obj = Named(name="Colony A")
    assert obj.name == "Colony A"


def test_named_setter_converts_value_to_str():
    obj = Named(name="x")
    obj.name = 123
    assert obj.name == "123"
    assert isinstance(obj.name, str)


# ---------------------------------------------------------------------------
# IdentifiedCollection
# ---------------------------------------------------------------------------

def test_collection_default_is_empty():
    coll = IdentifiedCollection()
    assert coll.count == 0
    assert coll.items == []


def test_collection_init_from_list_sorted_by_id():
    coll = IdentifiedCollection([Identified(id=3), Identified(id=1), Identified(id=2)])
    assert [item.id for item in coll.items] == [1, 2, 3]
    assert coll.count == 3


def test_collection_init_from_dict_uses_values():
    coll = IdentifiedCollection({2: Identified(id=2), 1: Identified(id=1)})
    assert [item.id for item in coll.items] == [1, 2]


def test_collection_init_copies_source_list():
    source = [Identified(id=1)]
    coll = IdentifiedCollection(source)
    source.append(Identified(id=2))
    assert coll.count == 1


@pytest.mark.xfail(
    reason="setter calls val.copy(), which tuples do not provide",
    raises=AttributeError,
    strict=True,
)
def test_collection_init_from_tuple():
    coll = IdentifiedCollection((Identified(id=2), Identified(id=1)))
    assert [item.id for item in coll.items] == [1, 2]


@pytest.mark.parametrize("bad_items", ["abc", 5, 3.14])
def test_collection_rejects_str_and_non_collections(bad_items):
    with pytest.raises(ValueError):
        IdentifiedCollection(bad_items)


def test_collection_items_returns_copy():
    coll = IdentifiedCollection([Identified(id=1), Identified(id=2)])
    snapshot = coll.items
    snapshot.clear()
    assert coll.count == 2


def test_add_creates_and_appends_identified_item():
    coll = IdentifiedCollection()
    item = coll.add(4)
    assert isinstance(item, Identified)
    assert item.id == 4
    assert coll.id_exists(4)
    assert coll.count == 1


def test_append_duplicate_id_raises():
    coll = IdentifiedCollection([Identified(id=1)])
    with pytest.raises(ValueError):
        coll.append(Identified(id=1))
    assert coll.count == 1


def test_exists_and_id_exists():
    coll = IdentifiedCollection([Identified(id=1), Identified(id=2)])
    assert coll.exists(Identified(id=2)) is True
    assert coll.exists(Identified(id=9)) is False
    assert coll.id_exists(1) is True
    assert coll.id_exists(9) is False


def test_get_item_found_and_missing():
    first = Identified(id=1)
    second = Identified(id=2)
    coll = IdentifiedCollection([first, second])
    assert coll.get_item(2) is second
    assert coll.get_item(1) is first
    assert coll.get_item(99) is None


def test_remove_existing_item_decreases_count():
    coll = IdentifiedCollection([Identified(id=1), Identified(id=2)])
    coll.remove(1)
    assert coll.id_exists(1) is False
    assert coll.id_exists(2) is True
    assert coll.count == 1


def test_remove_missing_item_raises_keyerror():
    coll = IdentifiedCollection([Identified(id=1)])
    with pytest.raises(KeyError):
        coll.remove(5)
    assert coll.count == 1


# ---------------------------------------------------------------------------
# Unique
# ---------------------------------------------------------------------------

def test_unique_is_identified_with_positive_auto_id():
    obj = Unique()
    assert isinstance(obj, Identified)
    assert obj.id > 0


def test_unique_ids_are_sequential():
    first = Unique()
    second = Unique()
    third = Unique()
    assert second.id == first.id + 1
    assert third.id == second.id + 1


def test_unique_id_setter_is_read_only():
    obj = Unique()
    original = obj.id
    obj.id = original + 1000
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
    ts = datetime(2024, 5, 1, 9, 30, 0)
    obj = TimeStamped(timestamp=ts)
    assert obj.timestamp == ts


def test_timestamped_setter_updates_timestamp():
    obj = TimeStamped(timestamp=datetime(2024, 1, 1))
    new_ts = datetime(2024, 2, 2)
    obj.timestamp = new_ts
    assert obj.timestamp == new_ts


# ---------------------------------------------------------------------------
# TimeStampElapsed
# ---------------------------------------------------------------------------

def test_timestamp_elapsed_defaults_initial_to_timestamp():
    ts = datetime(2024, 1, 1, 12, 0, 0)
    obj = TimeStampElapsed(timestamp=ts)
    assert obj.timestamp_initial == ts
    assert obj.timestamp_elapsed == timedelta(0)


def test_timestamp_elapsed_units():
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(hours=2, minutes=30, seconds=15)  # 9015 seconds
    obj = TimeStampElapsed(timestamp=t1, timestamp_initial=t0)

    assert obj.timestamp_elapsed == timedelta(hours=2, minutes=30, seconds=15)
    assert obj.timestamp_elapsed_seconds == pytest.approx(9015.0)
    assert obj.timestamp_elapsed_minutes == 150
    assert obj.timestamp_elapsed_hours == pytest.approx(9015 / 3600)


def test_timestamp_elapsed_minutes_truncates_toward_zero():
    t0 = datetime(2024, 1, 1)
    obj = TimeStampElapsed(timestamp=t0 + timedelta(seconds=59), timestamp_initial=t0)
    assert obj.timestamp_elapsed_minutes == 0


def test_timestamp_elapsed_updates_when_timestamp_changes():
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    obj = TimeStampElapsed(timestamp=t0, timestamp_initial=t0)
    obj.timestamp = t0 + timedelta(minutes=5)
    assert obj.timestamp_elapsed_seconds == pytest.approx(300.0)
    assert obj.timestamp_elapsed_minutes == 5
