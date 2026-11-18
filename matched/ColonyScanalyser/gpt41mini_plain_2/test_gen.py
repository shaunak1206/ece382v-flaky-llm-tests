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


def test_identified_valid_id():
    id_obj = Identified(5)
    assert id_obj.id == 5

def test_identified_invalid_id_type():
    with pytest.raises(ValueError):
        Identified("abc")

def test_identified_invalid_id_negative():
    with pytest.raises(ValueError):
        Identified(-3)

def test_id_exists_static_method():
    class Dummy(Identified):
        pass

    d1 = Dummy(1)
    d2 = Dummy(2)
    d3 = Dummy(3)
    collection = [d1, d2, d3]

    assert Identified._id_exists(collection, 2) is True
    assert Identified._id_exists(collection, 4) is False

def test_id_is_valid_static_method():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid(-1) is False
    assert Identified._id_is_valid("1") is False

def test_named_name_property_and_setter():
    n = Named("testname")
    assert n.name == "testname"
    n.name = 123
    assert n.name == "123"

def test_identifiedcollection_init_with_list_and_count():
    objs = [Identified(3), Identified(1), Identified(2)]
    coll = IdentifiedCollection(objs)
    assert coll.count == 3
    # items returns sorted list by id
    assert [item.id for item in coll.items] == [1, 2, 3]

def test_identifiedcollection_init_with_dict():
    objs = {1: Identified(10), 2: Identified(5)}
    coll = IdentifiedCollection(objs)
    assert coll.count == 2
    ids = [item.id for item in coll.items]
    assert 5 in ids and 10 in ids

def test_identifiedcollection_init_with_none_and_empty_items():
    coll = IdentifiedCollection()
    assert coll.count == 0
    coll.items = None
    assert coll.count == 0

def test_identifiedcollection_init_invalid_items():
    with pytest.raises(ValueError):
        IdentifiedCollection(123)

def test_identifiedcollection_add_method_creates_and_appends():
    coll = IdentifiedCollection()
    new_item = coll.add(7)
    assert isinstance(new_item, Identified)
    assert new_item.id == 7
    assert coll.id_exists(7) is True

def test_identifiedcollection_append_and_exists_id_exists():
    coll = IdentifiedCollection()
    obj = Identified(9)
    coll.append(obj)
    assert coll.exists(obj) is True
    with pytest.raises(ValueError):
        coll.append(obj)

def test_identifiedcollection_get_item_returns_correct_or_none():
    objs = [Identified(2), Identified(4)]
    coll = IdentifiedCollection(objs)
    found = coll.get_item(4)
    assert found is not None
    assert found.id == 4
    not_found = coll.get_item(10)
    assert not_found is None

def test_identifiedcollection_remove_existing_and_nonexistent():
    objs = [Identified(1), Identified(2)]
    coll = IdentifiedCollection(objs)
    coll.remove(1)
    assert coll.count == 1
    with pytest.raises(KeyError):
        coll.remove(100)

def test_unique_auto_increment_ids():
    Unique.id_count = 0  # reset class counter before test
    u1 = Unique()
    u2 = Unique()
    assert u1.id == 1
    assert u2.id == 2
    # test that id setter is disabled and does nothing
    u1.id = 100
    assert u1.id == 1  # unchanged

def test_timestamped_default_and_custom_timestamp():
    ts1 = TimeStamped()
    assert isinstance(ts1.timestamp, datetime)
    then = datetime(2020, 1, 1, 12, 0, 0)
    ts2 = TimeStamped(then)
    assert ts2.timestamp == then

def test_timestamp_elapsed_properties_accuracy():
    base_time = datetime(2020, 1, 1, 12, 0, 0)
    later_time = base_time + timedelta(hours=2, minutes=30, seconds=45)
    tse = TimeStampElapsed(timestamp=later_time, timestamp_initial=base_time)

    # timestamp_elapsed should be timedelta of 2h30m45s
    expected_td = later_time - base_time
    assert tse.timestamp_elapsed == expected_td

    # elapsed seconds
    assert abs(tse.timestamp_elapsed_seconds - expected_td.total_seconds()) < 0.0001

    # elapsed minutes (integer)
    assert tse.timestamp_elapsed_minutes == int(expected_td.total_seconds() // 60)

    # elapsed hours (float)
    expected_hours = expected_td.total_seconds() / 3600
    assert abs(tse.timestamp_elapsed_hours - expected_hours) < 1e-6

def test_timestamp_elapsed_defaults_to_now_and_initial():
    before = datetime.now() - timedelta(seconds=1)
    tse = TimeStampElapsed()
    after = datetime.now() + timedelta(seconds=1)
    # timestamp and timestamp_initial both default to roughly now
    assert before <= tse.timestamp <= after
    assert tse.timestamp_initial == tse.timestamp

def test_timestamp_initial_setter_and_getter():
    tse = TimeStampElapsed()
    new_init = datetime(2019, 5, 5, 5, 5, 5)
    tse.timestamp_initial = new_init
    assert tse.timestamp_initial == new_init
