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
    item = Identified(5)
    assert item.id == 5


@pytest.mark.parametrize("bad_id", [0, -1, 1.5, "3", None])
def test_identified_rejects_invalid_id(bad_id):
    with pytest.raises(ValueError):
        Identified(bad_id)


def test_identified_id_setter_validates():
    item = Identified(1)
    item.id = 10
    assert item.id == 10
    with pytest.raises(ValueError):
        item.id = -4
    assert item.id == 10


def test_identified_id_exists_static():
    items = [Identified(1), Identified(2)]
    assert Identified._id_exists(items, 2) is True
    assert Identified._id_exists(items, 3) is False
    assert Identified._id_exists([], 1) is False


# ---------- IdentifiedCollection ----------

def test_collection_default_is_empty():
    col = IdentifiedCollection()
    assert col.count == 0
    assert col.items == []


def test_collection_items_sorted_by_id():
    col = IdentifiedCollection([Identified(3), Identified(1), Identified(2)])
    assert [i.id for i in col.items] == [1, 2, 3]
    assert col.count == 3


def test_collection_accepts_dict_values():
    a, b = Identified(2), Identified(1)
    col = IdentifiedCollection({"x": a, "y": b})
    assert col.items == [b, a]


def test_collection_rejects_invalid_items_type():
    with pytest.raises(ValueError):
        IdentifiedCollection("abc")
    with pytest.raises(ValueError):
        IdentifiedCollection(5)


def test_collection_copies_input_and_output():
    source = [Identified(1)]
    col = IdentifiedCollection(source)
    source.append(Identified(2))
    assert col.count == 1

    returned = col.items
    returned.append(Identified(9))
    assert col.count == 1


def test_collection_add_creates_and_stores_item():
    col = IdentifiedCollection()
    item = col.add(4)
    assert isinstance(item, Identified)
    assert item.id == 4
    assert col.count == 1
    assert col.get_item(4) is item


def test_collection_add_duplicate_raises():
    col = IdentifiedCollection()
    col.add(1)
    with pytest.raises(ValueError):
        col.add(1)
    assert col.count == 1


def test_collection_append_and_exists():
    col = IdentifiedCollection()
    item = Identified(7)
    assert col.exists(item) is False
    col.append(item)
    assert col.exists(item) is True
    assert col.exists(Identified(7)) is True  # matches on id
    assert col.id_exists(7) is True
    assert col.id_exists(8) is False
    with pytest.raises(ValueError):
        col.append(Identified(7))


def test_collection_get_item_missing_returns_none():
    col = IdentifiedCollection([Identified(1)])
    assert col.get_item(99) is None
    assert col.get_item(1).id == 1


def test_collection_remove():
    col = IdentifiedCollection([Identified(1), Identified(2)])
    col.remove(1)
    assert col.count == 1
    assert col.id_exists(1) is False
    assert col.id_exists(2) is True


def test_collection_remove_missing_raises_keyerror():
    col = IdentifiedCollection([Identified(1)])
    with pytest.raises(KeyError):
        col.remove(5)
    assert col.count == 1


# ---------- Named ----------

def test_named_stores_and_converts_to_string():
    assert Named("colony").name == "colony"
    assert Named(42).name == "42"
    n = Named("a")
    n.name = 3.5
    assert n.name == "3.5"


# ---------- Unique ----------

def test_unique_ids_auto_increment():
    a = Unique()
    b = Unique()
    c = Unique()
    assert a.id > 0
    assert b.id == a.id + 1
    assert c.id == b.id + 1


def test_unique_id_is_read_only():
    u = Unique()
    original = u.id
    u.id = 12345
    assert u.id == original


# ---------- TimeStamped ----------

def test_timestamped_explicit_and_setter():
    ts = datetime(2020, 1, 2, 3, 4, 5)
    obj = TimeStamped(ts)
    assert obj.timestamp == ts
    new_ts = datetime(2021, 6, 7)
    obj.timestamp = new_ts
    assert obj.timestamp == new_ts


def test_timestamped_defaults_to_now():
    before = datetime.now()
    obj = TimeStamped()
    after = datetime.now()
    assert before <= obj.timestamp <= after


# ---------- TimeStampElapsed ----------

def test_elapsed_defaults_to_zero():
    obj = TimeStampElapsed()
    assert obj.timestamp_initial == obj.timestamp
    assert obj.timestamp_elapsed == timedelta(0)
    assert obj.timestamp_elapsed_seconds == 0


def test_elapsed_conversions():
    start = datetime(2020, 1, 1, 0, 0, 0)
    obj = TimeStampElapsed(start + timedelta(minutes=90), start)
    assert obj.timestamp_elapsed == timedelta(minutes=90)
    assert obj.timestamp_elapsed_seconds == 5400
    assert obj.timestamp_elapsed_minutes == 90
    assert obj.timestamp_elapsed_hours == pytest.approx(1.5)


def test_elapsed_minutes_truncates_and_updates_with_setters():
    start = datetime(2020, 1, 1)
    obj = TimeStampElapsed(start + timedelta(seconds=119), start)
    assert obj.timestamp_elapsed_minutes == 1

    obj.timestamp = start + timedelta(hours=2)
    assert obj.timestamp_elapsed_hours == pytest.approx(2.0)

    obj.timestamp_initial = start + timedelta(hours=1)
    assert obj.timestamp_elapsed == timedelta(hours=1)
