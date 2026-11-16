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


def make_items(*ids):
    return [Identified(id=i) for i in ids]


# ---------------------------------------------------------------------------
# Identified
# ---------------------------------------------------------------------------

def test_identified_stores_valid_id():
    assert Identified(5).id == 5


def test_identified_rejects_non_positive_ids():
    with pytest.raises(ValueError):
        Identified(0)
    with pytest.raises(ValueError):
        Identified(-3)


def test_identified_rejects_non_integer_ids():
    for bad in ("1", 1.0, None, [1]):
        with pytest.raises(ValueError):
            Identified(bad)


def test_id_setter_keeps_previous_value_when_invalid():
    obj = Identified(2)
    with pytest.raises(ValueError):
        obj.id = 0
    assert obj.id == 2


# ---------------------------------------------------------------------------
# IdentifiedCollection
# ---------------------------------------------------------------------------

def test_collection_default_is_empty():
    coll = IdentifiedCollection()
    assert coll.count == 0
    assert coll.items == []


def test_collection_items_sorted_by_id():
    coll = IdentifiedCollection(make_items(3, 1, 2))
    assert [i.id for i in coll.items] == [1, 2, 3]


def test_collection_copies_supplied_list():
    source = make_items(1, 2)
    coll = IdentifiedCollection(source)
    source.append(Identified(3))
    assert coll.count == 2


def test_collection_accepts_dict_and_uses_values():
    coll = IdentifiedCollection({"a": Identified(2), "b": Identified(1)})
    assert [i.id for i in coll.items] == [1, 2]


def test_collection_rejects_string_and_non_collection():
    with pytest.raises(ValueError):
        IdentifiedCollection("abc")
    with pytest.raises(ValueError):
        IdentifiedCollection(42)


def test_add_creates_and_appends_identified():
    coll = IdentifiedCollection()
    item = coll.add(7)
    assert isinstance(item, Identified)
    assert item.id == 7
    assert coll.get_item(7) is item


def test_append_duplicate_id_raises():
    coll = IdentifiedCollection(make_items(1))
    with pytest.raises(ValueError):
        coll.append(Identified(1))
    assert coll.count == 1


def test_exists_and_id_exists_match_by_id():
    coll = IdentifiedCollection([Identified(4)])
    assert coll.exists(Identified(4))
    assert coll.id_exists(4)
    assert not coll.id_exists(5)


def test_get_item_returns_match_or_none():
    coll = IdentifiedCollection(make_items(1, 2))
    assert coll.get_item(2).id == 2
    assert coll.get_item(99) is None


def test_remove_deletes_item():
    coll = IdentifiedCollection(make_items(1, 2, 3))
    coll.remove(2)
    assert [i.id for i in coll.items] == [1, 3]
    assert not coll.id_exists(2)


def test_remove_missing_id_raises_key_error():
    coll = IdentifiedCollection(make_items(1))
    with pytest.raises(KeyError):
        coll.remove(5)
    assert coll.count == 1


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="items setter calls .copy(), which tuple does not provide",
)
def test_collection_accepts_tuple():
    coll = IdentifiedCollection(tuple(make_items(1, 2)))
    assert coll.count == 2


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------

def test_named_stores_and_coerces_name_to_str():
    assert Named("colony").name == "colony"
    assert Named(12).name == "12"

    n = Named("a")
    n.name = "b"
    assert n.name == "b"


# ---------------------------------------------------------------------------
# Unique
# ---------------------------------------------------------------------------

def test_unique_is_identified_with_sequential_ids():
    first = Unique()
    second = Unique()
    assert isinstance(first, Identified)
    assert first.id > 0
    assert second.id == first.id + 1


def test_unique_id_is_read_only():
    u = Unique()
    original = u.id
    u.id = original + 100
    assert u.id == original


# ---------------------------------------------------------------------------
# TimeStamped
# ---------------------------------------------------------------------------

def test_timestamped_defaults_to_now():
    before = datetime.now()
    ts = TimeStamped()
    after = datetime.now()
    assert isinstance(ts.timestamp, datetime)
    assert before <= ts.timestamp <= after


def test_timestamped_stores_given_timestamp():
    when = datetime(2024, 1, 2, 3, 4, 5)
    assert TimeStamped(when).timestamp == when


# ---------------------------------------------------------------------------
# TimeStampElapsed
# ---------------------------------------------------------------------------

def test_elapsed_defaults_both_timestamps_to_now():
    before = datetime.now()
    tse = TimeStampElapsed()
    after = datetime.now()
    assert before <= tse.timestamp <= after
    assert tse.timestamp_initial == tse.timestamp
    assert tse.timestamp_elapsed == timedelta(0)


def test_elapsed_initial_defaults_to_timestamp():
    t = datetime(2024, 5, 1, 12, 0, 0)
    tse = TimeStampElapsed(timestamp=t)
    assert tse.timestamp_initial == t
    assert tse.timestamp_elapsed == timedelta(0)


def test_elapsed_is_difference_from_initial():
    start = datetime(2024, 5, 1, 12, 0, 0)
    now = start + timedelta(hours=1, minutes=30)
    tse = TimeStampElapsed(timestamp=now, timestamp_initial=start)
    assert tse.timestamp_elapsed == timedelta(hours=1, minutes=30)


def test_elapsed_seconds_minutes_and_hours():
    start = datetime(2024, 5, 1, 12, 0, 0)
    tse = TimeStampElapsed(
        timestamp=start + timedelta(seconds=5400),
        timestamp_initial=start,
    )
    assert tse.timestamp_elapsed_seconds == 5400
    assert tse.timestamp_elapsed_minutes == 90
    assert tse.timestamp_elapsed_hours == pytest.approx(1.5)


def test_elapsed_minutes_truncates_partial_minutes():
    start = datetime(2024, 5, 1, 12, 0, 0)
    tse = TimeStampElapsed(
        timestamp=start + timedelta(seconds=150),
        timestamp_initial=start,
    )
    assert tse.timestamp_elapsed_minutes == 2

    tse_short = TimeStampElapsed(
        timestamp=start + timedelta(seconds=59),
        timestamp_initial=start,
    )
    assert tse_short.timestamp_elapsed_minutes == 0


def test_elapsed_updates_when_timestamp_changes():
    start = datetime(2024, 1, 1, 0, 0, 0)
    tse = TimeStampElapsed(timestamp=start, timestamp_initial=start)
    tse.timestamp = start + timedelta(seconds=120)
    assert tse.timestamp_elapsed_minutes == 2
    assert isinstance(tse, TimeStamped)
