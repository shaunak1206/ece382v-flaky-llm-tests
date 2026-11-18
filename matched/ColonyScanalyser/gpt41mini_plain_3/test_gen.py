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


def test_identified_valid_id():
    obj = Identified(10)
    assert obj.id == 10


def test_identified_invalid_id_negative():
    with pytest.raises(ValueError):
        Identified(-1)


def test_identified_invalid_id_zero():
    with pytest.raises(ValueError):
        Identified(0)


def test_identified_invalid_id_non_int():
    with pytest.raises(ValueError):
        Identified("abc")


def test_id_exists_static_method_true_false():
    obj1 = Identified(1)
    obj2 = Identified(2)
    collection = [obj1, obj2]
    assert Identified._id_exists(collection, 1) is True
    assert Identified._id_exists(collection, 3) is False


def test_identifiedcollection_init_and_count():
    objs = [Identified(2), Identified(1)]
    col = IdentifiedCollection(objs)
    assert col.count == 2
    # Items should be sorted by id
    assert [item.id for item in col.items] == [1, 2]


def test_identifiedcollection_items_setter_with_dict():
    d = {1: Identified(1), 2: Identified(2)}
    col = IdentifiedCollection()
    col.items = d
    assert col.count == 2
    assert all(isinstance(i, Identified) for i in col.items)


def test_identifiedcollection_items_setter_none_and_invalid():
    col = IdentifiedCollection()
    col.items = None
    assert col.count == 0
    with pytest.raises(ValueError):
        col.items = "not a collection"


def test_identifiedcollection_add_and_exists_and_id_exists():
    col = IdentifiedCollection()
    item = col.add(5)
    assert isinstance(item, Identified)
    assert item.id == 5
    assert col.exists(item) is True
    assert col.id_exists(5) is True
    assert col.id_exists(999) is False


def test_identifiedcollection_append_duplicate_raises():
    col = IdentifiedCollection()
    item1 = Identified(1)
    col.append(item1)
    with pytest.raises(ValueError):
        col.append(item1)  # duplicate by id


def test_identifiedcollection_get_item():
    col = IdentifiedCollection()
    col.append(Identified(1))
    col.append(Identified(2))
    found = col.get_item(2)
    assert found is not None
    assert found.id == 2
    assert col.get_item(999) is None


def test_identifiedcollection_remove_existing_and_missing():
    col = IdentifiedCollection()
    col.append(Identified(3))
    col.remove(3)
    assert col.count == 0
    with pytest.raises(KeyError):
        col.remove(3)


def test_named_name_property():
    n = Named("testname")
    assert n.name == "testname"
    n.name = "changed"
    assert n.name == "changed"
    n.name = 123  # converts to string
    assert n.name == "123"


def test_unique_auto_incrementing_id(monkeypatch):
    # Reset id_count for testing
    Unique.id_count = 0
    u1 = Unique()
    u2 = Unique()
    assert u1.id == 1
    assert u2.id == 2
    # Attempting to set id does nothing (read-only)
    old_id = u1.id
    u1.id = 999  # Should do nothing
    assert u1.id == old_id


def test_timestamped_default_and_explicit_timestamp():
    ts1 = TimeStamped()
    # Roughly now, allow difference up to 1 sec
    assert isinstance(ts1.timestamp, datetime)
    now = datetime.now()
    delta = now - ts1.timestamp
    assert abs(delta.total_seconds()) < 1

    dt = datetime(2020, 1, 1, 12, 0)
    ts2 = TimeStamped(timestamp=dt)
    assert ts2.timestamp == dt


def test_timestamped_setter():
    ts = TimeStamped()
    new_time = datetime(2021, 1, 1, 1, 1)
    ts.timestamp = new_time
    assert ts.timestamp == new_time


def test_timestampelapsed_default_and_initial():
    now = datetime.now()
    elapsed = TimeStampElapsed()
    # timestamp and timestamp_initial should default roughly to now
    delta = elapsed.timestamp - elapsed.timestamp_initial
    assert isinstance(delta, timedelta)
    assert abs(delta.total_seconds()) < 1


def test_timestampelapsed_elapsed_properties():
    base_time = datetime(2020, 1, 1, 12, 0, 0)
    later_time = base_time + timedelta(hours=2, minutes=30, seconds=45)
    elapsed = TimeStampElapsed(timestamp=later_time, timestamp_initial=base_time)

    elapsed_td = elapsed.timestamp_elapsed
    assert elapsed_td == later_time - base_time

    assert pytest.approx(elapsed.timestamp_elapsed_hours, 0.01) == (2 + 30 / 60 + 45 / 3600)

    assert elapsed.timestamp_elapsed_minutes == int(elapsed.timestamp_elapsed_seconds / 60)

    # timestamp_elapsed_seconds returns float seconds, check close to 2*3600 + 30*60 + 45
    total_seconds_expected = 2 * 3600 + 30 * 60 + 45
    assert pytest.approx(elapsed.timestamp_elapsed_seconds, 0.01) == total_seconds_expected


def test_timestampelapsed_timestamp_initial_property_setter_getter():
    dt1 = datetime(2020, 1, 1)
    dt2 = datetime(2020, 1, 2)
    elapsed = TimeStampElapsed(timestamp=dt2, timestamp_initial=dt1)
    assert elapsed.timestamp_initial == dt1

    new_initial = datetime(2019, 12, 31)
    elapsed.timestamp_initial = new_initial
    assert elapsed.timestamp_initial == new_initial
