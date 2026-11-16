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
    assert Identified(5).id == 5


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1", None])
def test_identified_rejects_invalid_id_on_init(bad):
    with pytest.raises(ValueError):
        Identified(bad)


def test_identified_invalid_set_keeps_old_value():
    item = Identified(3)
    with pytest.raises(ValueError):
        item.id = -4
    assert item.id == 3
    item.id = 9
    assert item.id == 9


def test_identified_id_is_valid_and_exists_static_methods():
    assert Identified._id_is_valid(1) is True
    assert Identified._id_is_valid(0) is False
    assert Identified._id_is_valid("2") is False

    collection = [Identified(1), Identified(4)]
    assert Identified._id_exists(collection, 4) is True
    assert Identified._id_exists(collection, 2) is False
    assert Identified._id_exists([], 1) is False


# ---------- IdentifiedCollection ----------

def test_collection_defaults_to_empty():
    col = IdentifiedCollection()
    assert col.count == 0
    assert col.items == []


def test_collection_items_sorted_and_copied_from_input():
    source = [Identified(3), Identified(1), Identified(2)]
    col = IdentifiedCollection(source)
    assert [i.id for i in col.items] == [1, 2, 3]
    assert col.count == 3

    # Modifying the returned list or the source does not alter the collection
    col.items.pop()
    source.append(Identified(10))
    assert col.count == 3


def test_collection_accepts_dict_values():
    a, b = Identified(2), Identified(1)
    col = IdentifiedCollection({"x": a, "y": b})
    assert col.items == [b, a]


@pytest.mark.parametrize("bad", ["abc", 5, 3.2])
def test_collection_rejects_non_collection_items(bad):
    with pytest.raises(ValueError):
        IdentifiedCollection(bad)


def test_collection_add_creates_and_returns_item():
    col = IdentifiedCollection()
    item = col.add(7)
    assert isinstance(item, Identified)
    assert item.id == 7
    assert col.count == 1
    assert col.get_item(7) is item


def test_collection_append_duplicate_raises():
    col = IdentifiedCollection()
    col.append(Identified(1))
    with pytest.raises(ValueError):
        col.append(Identified(1))
    with pytest.raises(ValueError):
        col.add(1)
    assert col.count == 1


def test_collection_exists_and_id_exists():
    item = Identified(2)
    col = IdentifiedCollection([item])
    assert col.exists(item) is True
    assert col.exists(Identified(2)) is True  # matched by id, not identity
    assert col.exists(Identified(3)) is False
    assert col.id_exists(2) is True
    assert col.id_exists(99) is False


def test_collection_get_item_found_and_missing():
    a, b = Identified(1), Identified(2)
    col = IdentifiedCollection([a, b])
    assert col.get_item(2) is b
    assert col.get_item(5) is None


def test_collection_remove_existing_and_missing():
    a, b = Identified(1), Identified(2)
    col = IdentifiedCollection([a, b])
    col.remove(1)
    assert col.items == [b]
    assert col.id_exists(1) is False
    with pytest.raises(KeyError):
        col.remove(1)
    with pytest.raises(KeyError):
        col.remove(42)


# ---------- Named ----------

def test_named_stores_string_and_coerces_non_strings():
    n = Named("colony")
    assert n.name == "colony"
    n.name = 123
    assert n.name == "123"
    assert isinstance(n.name, str)


# ---------- Unique ----------

def test_unique_ids_auto_increment_and_are_distinct():
    a = Unique()
    b = Unique()
    c = Unique()
    assert b.id == a.id + 1
    assert c.id == b.id + 1
    assert len({a.id, b.id, c.id}) == 3
    assert Unique.id_count == c.id


def test_unique_id_is_read_only_and_works_in_collection():
    u = Unique()
    original = u.id
    u.id = 9999
    assert u.id == original

    col = IdentifiedCollection([u])
    assert col.exists(u)
    assert col.get_item(original) is u


# ---------- TimeStamped ----------

def test_timestamped_default_is_now_and_explicit_is_kept():
    before = datetime.now()
    ts = TimeStamped()
    after = datetime.now()
    assert before <= ts.timestamp <= after

    fixed = datetime(2020, 1, 2, 3, 4, 5)
    ts2 = TimeStamped(fixed)
    assert ts2.timestamp == fixed
    new = datetime(2021, 6, 7)
    ts2.timestamp = new
    assert ts2.timestamp == new


# ---------- TimeStampElapsed ----------

def test_elapsed_defaults_to_zero_when_no_initial():
    fixed = datetime(2020, 1, 1, 12, 0, 0)
    ts = TimeStampElapsed(fixed)
    assert ts.timestamp_initial == fixed
    assert ts.timestamp_elapsed == timedelta(0)
    assert ts.timestamp_elapsed_seconds == 0
    assert ts.timestamp_elapsed_minutes == 0
    assert ts.timestamp_elapsed_hours == 0

    now_ts = TimeStampElapsed()
    assert now_ts.timestamp == now_ts.timestamp_initial


def test_elapsed_values_are_computed_correctly():
    initial = datetime(2020, 1, 1, 0, 0, 0)
    ts = TimeStampElapsed(initial + timedelta(hours=1, minutes=30), initial)
    assert ts.timestamp_elapsed == timedelta(hours=1, minutes=30)
    assert ts.timestamp_elapsed_seconds == 5400
    assert ts.timestamp_elapsed_minutes == 90
    assert ts.timestamp_elapsed_hours == pytest.approx(1.5)


def test_elapsed_minutes_truncates_and_updates_with_timestamp():
    initial = datetime(2020, 1, 1)
    ts = TimeStampElapsed(initial + timedelta(seconds=119), initial)
    assert ts.timestamp_elapsed_minutes == 1

    ts.timestamp = initial + timedelta(hours=2)
    assert ts.timestamp_elapsed_hours == pytest.approx(2.0)

    ts.timestamp_initial = initial + timedelta(hours=3)
    assert ts.timestamp_elapsed_seconds == -3600
    assert ts.timestamp_elapsed_hours == pytest.approx(-1.0)
