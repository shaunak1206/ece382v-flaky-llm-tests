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
    obj = Identified(5)
    assert obj.id == 5


@pytest.mark.parametrize("invalid_id", [-1, 0, 3.5, "a", None])
def test_identified_invalid_id_raises(invalid_id):
    with pytest.raises(ValueError):
        Identified(invalid_id)


def test_identified__id_exists_true_false():
    objs = [Identified(1), Identified(2), Identified(3)]
    assert Identified._id_exists(objs, 2)
    assert not Identified._id_exists(objs, 4)


def test_identified__id_is_valid():
    assert Identified._id_is_valid(1)
    assert not Identified._id_is_valid(0)
    assert not Identified._id_is_valid(-10)
    assert not Identified._id_is_valid("1")


def test_identifiedcollection_init_and_items_property():
    objs = [Identified(3), Identified(1), Identified(2)]
    collection = IdentifiedCollection(objs)
    # Items should be sorted by id
    sorted_ids = [item.id for item in collection.items]
    assert sorted_ids == [1, 2, 3]
    # Original items is a copy
    assert collection.items is not objs


def test_identifiedcollection_count_property():
    collection = IdentifiedCollection([Identified(1), Identified(2)])
    assert collection.count == 2
    collection_empty = IdentifiedCollection()
    assert collection_empty.count == 0


def test_identifiedcollection_items_setter_accepts_collections():
    collection = IdentifiedCollection()
    collection.items = [Identified(1)]
    assert collection.count == 1
    collection.items = (Identified(2), Identified(3))
    assert collection.count == 2
    collection.items = {4: Identified(4), 5: Identified(5)}
    ids = [item.id for item in collection.items]
    assert set(ids) == {4, 5}
    collection.items = []
    assert collection.count == 0
    with pytest.raises(ValueError):
        collection.items = 123
    with pytest.raises(ValueError):
        collection.items = "not a collection"
    collection.items = None
    assert collection.count == 0


def test_identifiedcollection_add_and_append_and_exists_and_id_exists():
    collection = IdentifiedCollection()
    item = collection.add(10)
    assert isinstance(item, Identified)
    assert item.id == 10
    assert collection.exists(item)
    assert collection.id_exists(10)
    # append with duplicate id raises
    with pytest.raises(ValueError):
        collection.append(item)


def test_identifiedcollection_get_item():
    collection = IdentifiedCollection([Identified(1), Identified(2)])
    item = collection.get_item(1)
    assert item is not None
    assert item.id == 1
    assert collection.get_item(10) is None


def test_identifiedcollection_remove():
    collection = IdentifiedCollection([Identified(1), Identified(2)])
    collection.remove(1)
    assert collection.get_item(1) is None
    with pytest.raises(KeyError):
        collection.remove(10)


def test_named_property_and_setter():
    n = Named("TestName")
    assert n.name == "TestName"
    n.name = "NewName"
    assert n.name == "NewName"
    n.name = 123
    assert n.name == "123"  # converted to string


def test_unique_id_auto_increment():
    # Reset counter for deterministic test
    Unique.id_count = 0
    u1 = Unique()
    u2 = Unique()
    assert u1.id == 1
    assert u2.id == 2
    # id setter is overridden to no-op, so setting id does nothing
    old_id = u1.id
    u1.id = 100  # should not change id
    assert u1.id == old_id


def test_timestamped_default_and_custom_timestamp():
    now = datetime.now()
    ts1 = TimeStamped()
    # timestamp will be close to now; allow small delta
    assert isinstance(ts1.timestamp, datetime)
    ts2 = TimeStamped(timestamp=now)
    assert ts2.timestamp == now
    new_time = datetime(2020, 1, 1)
    ts2.timestamp = new_time
    assert ts2.timestamp == new_time


def test_timestampelapsed_properties_elapsed_times():
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    later_time = base_time + timedelta(hours=2, minutes=30, seconds=10)
    tse = TimeStampElapsed(timestamp=later_time, timestamp_initial=base_time)
    assert tse.timestamp == later_time
    assert tse.timestamp_initial == base_time

    elapsed = tse.timestamp_elapsed
    assert isinstance(elapsed, timedelta)
    assert elapsed.total_seconds() == (2 * 3600 + 30 * 60 + 10)

    # timestamp_elapsed_seconds returns total_seconds as int (via property)
    # Actually returns float from total_seconds(), as per code, test type and approximate value
    secs = tse.timestamp_elapsed_seconds
    assert abs(secs - elapsed.total_seconds()) < 0.001

    # timestamp_elapsed_minutes returns int minutes truncating seconds
    expected_minutes = int(elapsed.total_seconds() / 60)
    assert tse.timestamp_elapsed_minutes == expected_minutes

    # timestamp_elapsed_hours returns float hours (seconds/60/60)
    expected_hours = (elapsed.total_seconds() / 60) / 60
    assert abs(tse.timestamp_elapsed_hours - expected_hours) < 0.0001


def test_timestampelapsed_defaults_to_now(monkeypatch):
    fixed_now = datetime(2022, 2, 2, 2, 2, 2)
    monkeypatch.setattr("colonyscanalyser.base.datetime", FixedDateTime(fixed_now))
    tse = TimeStampElapsed()
    assert tse.timestamp == fixed_now
    assert tse.timestamp_initial == fixed_now


class FixedDateTime(datetime):
    # Helper to monkeypatch datetime.now()
    _fixed_now = None

    def __new__(cls, fixed_now):
        cls._fixed_now = fixed_now
        return datetime.__new__(datetime)

    @classmethod
    def now(cls, tz=None):
        return cls._fixed_now


def test_identifiedcollection_append_only_if_not_exists():
    collection = IdentifiedCollection()
    item1 = Identified(1)
    collection.append(item1)
    assert collection.count == 1
    with pytest.raises(ValueError):
        collection.append(Identified(1))  # duplicate id


def test_identifiedcollection_items_returns_sorted_copy():
    collection = IdentifiedCollection([Identified(5), Identified(3), Identified(4)])
    sorted_ids = [item.id for item in collection.items]
    assert sorted_ids == [3, 4, 5]
    # modifying the returned list does not affect internal _items
    items_copy = collection.items
    items_copy.append(Identified(10))
    assert collection.count == 3


def test_unique_id_count_increment_affects_new_instances():
    Unique.id_count = 10
    u = Unique()
    assert u.id == 11
    u2 = Unique()
    assert u2.id == 12


def test_identifiedcollection_remove_removes_correct_item():
    items = [Identified(1), Identified(2), Identified(3)]
    collection = IdentifiedCollection(items)
    collection.remove(2)
    ids = [item.id for item in collection.items]
    assert 2 not in ids
    assert len(ids) == 2
    # removing again causes KeyError
    with pytest.raises(KeyError):
        collection.remove(2)
