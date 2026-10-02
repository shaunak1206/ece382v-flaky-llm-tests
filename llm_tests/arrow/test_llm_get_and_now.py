"""Unit tests for arrow.get, arrow.now, arrow.utcnow and the Arrow constructors."""
import time
from datetime import date, datetime, timedelta, timezone

import pytest
from dateutil import tz

import arrow
from arrow import Arrow


def test_get_no_args_returns_current_utc():
    before = datetime.now(timezone.utc)
    result = arrow.get()
    after = datetime.now(timezone.utc)
    assert before <= result.datetime <= after
    assert result.utcoffset() == timedelta(0)


def test_utcnow_is_utc_and_close_to_system_time():
    result = arrow.utcnow()
    assert isinstance(result, Arrow)
    assert result.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - result.datetime).total_seconds()) < 5


def test_now_defaults_to_local_timezone():
    result = arrow.now()
    local_offset = datetime.now().astimezone().utcoffset()
    assert result.utcoffset() == local_offset
    assert abs(result.timestamp() - time.time()) < 5


def test_now_with_timezone_string():
    result = arrow.now("US/Pacific")
    assert result.utcoffset() in (timedelta(hours=-7), timedelta(hours=-8))
    assert abs(result.timestamp() - time.time()) < 5


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "1970-01-01T00:00:00+00:00"),
        (1367992474, "2013-05-08T05:54:34+00:00"),
        (1367992474.293378, "2013-05-08T05:54:34.293378+00:00"),
        ("1367992474", "2013-05-08T05:54:34+00:00"),
    ],
)
def test_get_from_timestamp(value, expected):
    if isinstance(value, str):
        result = arrow.get(float(value))
    else:
        result = arrow.get(value)
    assert result.isoformat() == expected


def test_get_from_timestamp_with_tzinfo():
    result = arrow.get(1367992474, tzinfo="US/Pacific")
    assert result.isoformat() == "2013-05-07T22:54:34-07:00"


def test_get_from_naive_datetime_assumes_utc():
    result = arrow.get(datetime(2013, 5, 5, 12, 30))
    assert result.isoformat() == "2013-05-05T12:30:00+00:00"


def test_get_from_aware_datetime_keeps_timezone():
    dt = datetime(2013, 5, 5, 12, 30, tzinfo=tz.gettz("US/Pacific"))
    result = arrow.get(dt)
    assert result.isoformat() == "2013-05-05T12:30:00-07:00"


def test_get_from_arrow_copies_value():
    original = arrow.get(2020, 1, 2, 3, 4, 5)
    copy = arrow.get(original)
    assert copy == original
    assert copy is not original


def test_get_from_integers():
    result = arrow.get(2013, 5, 5, 12, 30, 45)
    assert result.isoformat() == "2013-05-05T12:30:45+00:00"


def test_get_datetime_with_replacement_timezone():
    result = arrow.get(datetime(2013, 5, 5), "US/Pacific")
    assert result.isoformat() == "2013-05-05T00:00:00-07:00"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-09-29T01:26:43.830580", "2013-09-29T01:26:43.830580+00:00"),
        ("2013-09-29", "2013-09-29T00:00:00+00:00"),
        ("2013-09-29T01:26:43+02:00", "2013-09-29T01:26:43+02:00"),
        ("2013-09-29T01:26:43Z", "2013-09-29T01:26:43+00:00"),
        ("20160413T133656.456289", "2016-04-13T13:36:56.456289+00:00"),
        ("2013-W05-4", "2013-01-31T00:00:00+00:00"),
    ],
)
def test_get_parses_iso_strings(text, expected):
    assert arrow.get(text).isoformat() == expected


def test_get_with_format_string():
    result = arrow.get("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss")
    assert result.isoformat() == "2013-05-05T12:30:45+00:00"


def test_get_with_format_list():
    result = arrow.get("2013-05-05 12:30:45", ["MM/DD/YYYY", "YYYY-MM-DD HH:mm:ss"])
    assert result == arrow.Arrow(2013, 5, 5, 12, 30, 45)


def test_get_none_raises_type_error():
    with pytest.raises(TypeError):
        arrow.get(None)


def test_get_invalid_string_raises_value_error():
    with pytest.raises(ValueError):
        arrow.get("not a date")


def test_factory_with_subclass_returns_subclass():
    class MyArrow(Arrow):
        pass

    factory = arrow.api.factory(MyArrow)
    assert isinstance(factory.get(2020, 1, 1), MyArrow)
    assert isinstance(factory.utcnow(), MyArrow)


def test_arrow_constructor_defaults_to_utc():
    a = Arrow(2020, 2, 29)
    assert a.tzinfo == timezone.utc
    assert (a.hour, a.minute, a.second, a.microsecond) == (0, 0, 0, 0)


def test_fromtimestamp_and_utcfromtimestamp():
    assert Arrow.utcfromtimestamp(86400).isoformat() == "1970-01-02T00:00:00+00:00"
    assert Arrow.fromtimestamp(86400, tzinfo="Asia/Tokyo").isoformat() == (
        "1970-01-02T09:00:00+09:00"
    )


def test_strptime_and_fromordinal():
    a = Arrow.strptime("2019-01-20 15:49:10", "%Y-%m-%d %H:%M:%S")
    assert a.isoformat() == "2019-01-20T15:49:10+00:00"
    assert Arrow.fromordinal(737741).isoformat() == "2020-11-12T00:00:00+00:00"
