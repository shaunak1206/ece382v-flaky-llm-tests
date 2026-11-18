import pytest
from datetime import datetime, timedelta
from src.colonyscanalyser.base import (
    Identified,
    IdentifiedCollection,
    Named,
    Unique,
    TimeStamped,
    TimeStampElapsed,
)


def test_identified_id_property_and_validation():
    obj = Identified(5)
    assert obj.id == 5

    obj.id = 10
    assert obj.id == 10

    with pytest.raises(ValueError):
        obj.id = -1
    with pytest.raises(ValueError):
        obj.id = 0
    with pytest.raises(ValueError):
        obj.id = "string"
    with pytest.raises(ValueError):
        Identified._id_is_valid(0) is False
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(100) is True


def test_id_exists_static_method():
    class Dummy:
        def __init__(self, i):
            self.id = i

    d1 = Dummy(1)
    d2 = Dummy(2)
    collection = [d1, d2]

    assert Identified._id_exists(collection, 1) is True
    assert Identified._id_exists(collection, 3) is False


def test_named_name_property():
    n = Named("testname")
    assert n.name == "testname"
    n.name = "changed"
    assert n.name == "changed"
    n.name = 123
    assert n.name == "123"  # name converts to string


def test_unique_auto_increment_and_id_read_only():
    # Reset counter
    Unique.id_count = 0

    u1 = Unique()
    assert u1.id == 1
    u2 = Unique()
    assert u2.id == 2
    # Attempt to set id does nothing
    u1.id = 100
    assert u1.id == 1
    u2.id = -5
    assert u2.id == 2


def test_identifiedcollection_items_setter_accepts_collections_and_dicts():
    obj1 = Identified(1)
    obj2 = Identified(2)
    coll = IdentifiedCollection()
    coll.items = [obj1, obj2]
    # Should sort by id ascending
    result = coll.items
    assert result[0].id == 1
    assert result[1].id == 2

    coll.items = {10: obj1, 20: obj2}
    result = coll.items
    assert isinstance(result, list)
    assert sorted(i.id for i in result) == [1, 2]

    coll.items = None
    assert coll.items == []

    with pytest.raises(ValueError):
        coll.items = "not a collection"


def test_identifiedcollection_add_and_append_and_exists_id_exists_get_item():
    coll = IdentifiedCollection()
    # add returns Identified object with given id
    obj1 = coll.add(3)
    assert isinstance(obj1, Identified)
    assert obj1.id == 3
    assert coll.count == 1

    # append existing item with new id works
    obj2 = Identified(4)
    coll.append(obj2)
    assert coll.count == 2

    # append duplicate id raises
    obj_dup = Identified(3)
    with pytest.raises(ValueError):
        coll.append(obj_dup)

    assert coll.exists(obj1) is True
    assert coll.exists(Identified(999)) is False

    assert coll.id_exists(3) is True
    assert coll.id_exists(999) is False

    getobj = coll.get_item(3)
    assert getobj.id == 3
    assert coll.get_item(999) is None


def test_identifiedcollection_remove_item():
    coll = IdentifiedCollection()
    coll.items = [Identified(1), Identified(2), Identified(3)]
    coll.remove(2)
    assert coll.count == 2
    ids = [item.id for item in coll.items]
    assert 2 not in ids

    with pytest.raises(KeyError):
        coll.remove(999)

    # Removing last
    coll.remove(1)
    assert coll.count == 1
    assert coll.items[0].id == 3


def test_timestamped_defaults_and_setter_getter():
    before = datetime.now() - timedelta(seconds=1)
    ts = TimeStamped()
    after = datetime.now() + timedelta(seconds=1)
    assert before <= ts.timestamp <= after

    new_time = datetime(2020, 1, 1, 12, 0, 0)
    ts.timestamp = new_time
    assert ts.timestamp == new_time


def test_timestampelapsed_basic_elapsed_properties():
    t0 = datetime(2022, 1, 1, 10, 0, 0)
    t1 = datetime(2022, 1, 1, 12, 15, 30)
    elapsed_obj = TimeStampElapsed(timestamp=t1, timestamp_initial=t0)

    elapsed = t1 - t0
    assert elapsed_obj.timestamp_elapsed == elapsed
    # elapsed in seconds with fractional
    seconds = elapsed.total_seconds()
    # Property timestamp_elapsed_seconds returns total_seconds as float
    assert abs(elapsed_obj.timestamp_elapsed_seconds - seconds) < 1e-6

    # elapsed minutes, integer division
    expected_minutes = int(seconds // 60)
    assert elapsed_obj.timestamp_elapsed_minutes == expected_minutes

    # elapsed hours = minutes / 60 (float)
    expected_hours = expected_minutes / 60
    assert abs(elapsed_obj.timestamp_elapsed_hours - expected_hours) < 1e-6


def test_timestampelapsed_initial_and_current_timestamp_defaults_and_setters():
    # Without args, timestamp_initial == timestamp
    obj = TimeStampElapsed()
    assert isinstance(obj.timestamp, datetime)
    assert obj.timestamp == obj.timestamp_initial

    new_initial = datetime(2020, 5, 5, 5, 5, 5)
    obj.timestamp_initial = new_initial
    assert obj.timestamp_initial == new_initial
    # timestamp_elapsed now = timestamp - new_initial
    expected_timedelta = obj.timestamp - new_initial
    assert obj.timestamp_elapsed == expected_timedelta


def test_identifiedcollection_items_returns_sorted_copy():
    coll = IdentifiedCollection()
    objs = [Identified(100), Identified(10), Identified(50)]
    coll.items = objs

    items1 = coll.items
    items2 = coll.items
    # Sorted by id ascending
    ids = [it.id for it in items1]
    assert ids == sorted(ids)

    # Items is a copy (safe from external modification)
    items1.append(Identified(0))
    assert len(coll.items) == len(objs)


def test_identified_collection_count_property_matches_items_length():
    coll = IdentifiedCollection()
    coll.items = []
    assert coll.count == 0

    coll.items = [Identified(1)]
    assert coll.count == 1

    coll.items = [Identified(i) for i in range(1, 10)]
    assert coll.count == 9


def test_unique_id_count_persists_across_instances():
    Unique.id_count = 0
    u1 = Unique()
    u2 = Unique()
    assert u1.id == 1
    assert u2.id == 2
    Unique.id_count = 5
    u3 = Unique()
    assert u3.id == 6


def test_identified_id_setter_rejects_invalid_types():
    with pytest.raises(ValueError):
        Identified(-10)
    with pytest.raises(ValueError):
        Identified(0)
    with pytest.raises(ValueError):
        Identified("a string")


def test_identifiedcollection_add_duplicates_raises():
    coll = IdentifiedCollection()
    coll.add(1)
    with pytest.raises(ValueError):
        coll.add(1)


def test_append_existing_item_not_in_collection_adds():
    coll = IdentifiedCollection()
    obj1 = Identified(99)
    # Initially empty
    coll.append(obj1)
    assert coll.count == 1
    assert coll.items[0].id == 99


def test_remove_item_removes_only_item():
    coll = IdentifiedCollection()
    coll.items = [Identified(1)]
    coll.remove(1)
    assert coll.count == 0
    with pytest.raises(KeyError):
        coll.remove(1)
