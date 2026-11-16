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


# ---------- Identified ----------

def test_identified_stores_valid_id():
    obj = Identified(5)
    assert obj.id == 5


@pytest.mark.parametrize("bad_id", [0, -1, 1.5, "3", None])
def test_identified_rejects_invalid_id_on_init(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_identified_id_setter_validates_and_keeps_old_value():
    obj = Identified(2)
    obj.id = 7
    assert obj.id == 7
    with pytest.raises(ValueError):
        obj.id = -4
    assert obj.id == 7


def test_identified_id_exists_static():
    items = [Identified(1), Identified(3)]
    assert Identified._id_exists(items, 3) is True
    assert Identified._id_exists(items, 2) is False
    assert Identified._id_exists([], 1) is False


# ---------- IdentifiedCollection ----------

def test_collection_default_is_empty():
    coll = IdentifiedCollection()
    assert coll.count == 0
    assert coll.items == []


def test_collection_items_sorted_by_id_and_copy_returned():
    a, b, c = Identified(3), Identified(1), Identified(2)
    coll = IdentifiedCollection([a, b, c])
    assert [i.id for i in coll.items] == [1, 2, 3]
    assert coll.count == 3

    # Modifying the returned list does not affect the collection
    coll.items.append(Identified(10))
    assert coll.count == 3


def test_collection_init_copies_input_list():
    source = [Identified(1)]
    coll = IdentifiedCollection(source)
    source.append(Identified(2))
    assert coll.count == 1


def test_collection_accepts_dict_values():
    a, b = Identified(2), Identified(1)
    coll = IdentifiedCollection({"x": a, "y": b})
    assert coll.items == [b, a]


@pytest.mark.parametrize("bad", ["abc", 5, 3.2])
def test_collection_rejects_non_collections(bad):
    with pytest.raises(ValueError):
        IdentifiedCollection(bad)


def test_collection_add_creates_and_returns_item():
    coll = IdentifiedCollection()
    item = coll.add(4)
    assert isinstance(item, Identified)
    assert item.id == 4
    assert coll.count == 1
    assert coll.get_item(4) is item


def test_collection_add_duplicate_raises():
    coll = IdentifiedCollection()
    coll.add(1)
    with pytest.raises(ValueError):
        coll.add(1)
    assert coll.count == 1


def test_collection_add_invalid_id_raises():
    coll = IdentifiedCollection()
    with pytest.raises(ValueError):
        coll.add(0)
    assert coll.count == 0


def test_collection_append_and_exists():
    coll = IdentifiedCollection()
    item = Identified(8)
    assert coll.exists(item) is False
    coll.append(item)
    assert coll.exists(item) is True
    assert coll.id_exists(8) is True
    assert coll.id_exists(9) is False
    # A different object with the same id counts as existing
    with pytest.raises(ValueError):
        coll.append(Identified(8))


def test_collection_get_item_missing_returns_none():
    coll = IdentifiedCollection([Identified(1)])
    assert coll.get_item(1).id == 1
    assert coll.get_item(99) is None


def test_collection_remove_existing_and_missing():
    coll = IdentifiedCollection([Identified(1), Identified(2), Identified(3)])
    coll.remove(2)
    assert [i.id for i in coll.items] == [1, 3]
    assert coll.id_exists(2) is False
    with pytest.raises(KeyError):
        coll.remove(2)


# ---------- Named ----------

def test_named_stores_and_converts_to_string():
    n = Named("colony")
    assert n.name == "colony"
    n.name = 42
    assert n.name == "42"
    assert isinstance(n.name, str)


# ---------- Unique ----------

def test_unique_ids_auto_increment():
    first = Unique()
    second = Unique()
    third = Unique()
    assert second.id == first.id + 1
    assert third.id == second.id + 1
    assert Unique.id_count >= third.id


def test_unique_id_is_read_only():
    u = Unique()
    original = u.id
    u.id = original + 100
    assert u.id == original


def test_unique_works_in_collection():
    a, b = Unique(), Unique()
    coll = IdentifiedCollection([b, a])
    assert coll.items == [a, b]
    assert coll.get_item(b.id) is b


# ---------- TimeStamped ----------

def test_timestamped_defaults_to_now():
    before = datetime.now()
    ts = TimeStamped()
    after = datetime.now()
    assert before <= ts.timestamp <= after


def test_timestamped_explicit_and_setter():
    t1 = datetime(2020, 1, 1, 12, 0, 0)
    t2 = datetime(2021, 6, 15, 8, 30, 0)
    ts = TimeStamped(t1)
    assert ts.timestamp == t1
    ts.timestamp = t2
    assert ts.timestamp == t2


# ---------- TimeStampElapsed ----------

def test_elapsed_defaults_initial_to_timestamp():
    t = datetime(2020, 1, 1, 12, 0, 0)
    ts = TimeStampElapsed(t)
    assert ts.timestamp == t
    assert ts.timestamp_initial == t
    assert ts.timestamp_elapsed == timedelta(0)
    assert ts.timestamp_elapsed_seconds == 0
    assert ts.timestamp_elapsed_minutes == 0
    assert ts.timestamp_elapsed_hours == 0


def test_elapsed_default_timestamp_is_now():
    before = datetime.now()
    ts = TimeStampElapsed()
    after = datetime.now()
    assert before <= ts.timestamp <= after
    assert ts.timestamp_initial == ts.timestamp


def test_elapsed_calculations():
    initial = datetime(2020, 1, 1, 0, 0, 0)
    current = datetime(2020, 1, 1, 1, 30, 30)
    ts = TimeStampElapsed(current, initial)
    assert ts.timestamp_elapsed == timedelta(hours=1, minutes=30, seconds=30)
    assert ts.timestamp_elapsed_seconds == 5430
    assert ts.timestamp_elapsed_minutes == 90
    assert ts.timestamp_elapsed_hours == pytest.approx(5430 / 3600)


def test_elapsed_updates_when_timestamps_change():
    initial = datetime(2020, 1, 1, 0, 0, 0)
    ts = TimeStampElapsed(datetime(2020, 1, 1, 2, 0, 0), initial)
    assert ts.timestamp_elapsed_hours == pytest.approx(2.0)
    ts.timestamp = datetime(2020, 1, 1, 3, 0, 0)
    assert ts.timestamp_elapsed_hours == pytest.approx(3.0)
    ts.timestamp_initial = datetime(2020, 1, 1, 2, 0, 0)
    assert ts.timestamp_elapsed_hours == pytest.approx(1.0)


def test_elapsed_negative_when_timestamp_before_initial():
    initial = datetime(2020, 1, 1, 1, 0, 0)
    current = datetime(2020, 1, 1, 0, 0, 0)
    ts = TimeStampElapsed(current, initial)
    assert ts.timestamp_elapsed == timedelta(hours=-1)
    assert ts.timestamp_elapsed_minutes == -60
