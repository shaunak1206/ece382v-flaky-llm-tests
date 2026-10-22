"""Tests for arrow.get / arrow.now / arrow.utcnow / arrow.factory (api.py and factory.py)."""

import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from freezegun import freeze_time

import arrow
from arrow import Arrow
from arrow.factory import ArrowFactory

UTC = timezone.utc


@freeze_time("2024-03-05 10:20:30")
def test_get_no_args_returns_current_utc_time():
    result = arrow.get()
    assert result == Arrow(2024, 3, 5, 10, 20, 30)
    assert result.utcoffset() == timedelta(0)


@freeze_time("2024-03-05 12:00:00")
@pytest.mark.parametrize(
    "tz, expected",
    [
        ("+09:00", "2024-03-05T21:00:00+09:00"),
        (timezone(timedelta(hours=-5)), "2024-03-05T07:00:00-05:00"),
    ],
)
def test_get_no_args_with_tzinfo_converts_now(tz, expected):
    assert arrow.get(tzinfo=tz).isoformat() == expected


@freeze_time("2024-03-05 10:20:30")
def test_utcnow_matches_frozen_time():
    result = arrow.utcnow()
    assert isinstance(result, Arrow)
    assert result.isoformat() == "2024-03-05T10:20:30+00:00"


@freeze_time("2024-03-05 12:00:00")
@pytest.mark.parametrize(
    "tz, expected",
    [
        ("+02:00", "2024-03-05T14:00:00+02:00"),
        (timezone(timedelta(hours=-3)), "2024-03-05T09:00:00-03:00"),
        ("utc", "2024-03-05T12:00:00+00:00"),
    ],
)
def test_now_with_timezone_expression(tz, expected):
    assert arrow.now(tz).isoformat() == expected


def test_now_with_invalid_timezone_raises():
    with pytest.raises(ValueError):
        arrow.now("Not/A_Zone_At_All")


@pytest.mark.parametrize(
    "value, kwargs, expected",
    [
        (1367992474, {}, "2013-05-08T05:54:34+00:00"),
        (1367992474.5, {}, "2013-05-08T05:54:34.500000+00:00"),
        (Decimal("1367992474"), {}, "2013-05-08T05:54:34+00:00"),
        (1367992474, {"tzinfo": "+02:00"}, "2013-05-08T07:54:34+02:00"),
    ],
)
def test_get_numeric_timestamps(value, kwargs, expected):
    assert arrow.get(value, **kwargs).isoformat() == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-09-29T01:26:43.830580", "2013-09-29T01:26:43.830580+00:00"),
        ("20160413T133656.456289", "2016-04-13T13:36:56.456289+00:00"),
        ("2013-05-05T12:30:45+02:00", "2013-05-05T12:30:45+02:00"),
        ("2013-05-05", "2013-05-05T00:00:00+00:00"),
    ],
)
def test_get_iso_strings(text, expected):
    assert arrow.get(text).isoformat() == expected


def test_get_string_options_tzinfo_and_whitespace():
    replaced = arrow.get("2013-05-05 12:30:45", tzinfo="+09:00")
    assert replaced.isoformat() == "2013-05-05T12:30:45+09:00"
    normalized = arrow.get("2013-05-05   12:30:45", normalize_whitespace=True)
    assert normalized == Arrow(2013, 5, 5, 12, 30, 45)


def test_get_datetime_naive_is_utc_and_aware_keeps_timezone():
    naive = arrow.get(datetime(2013, 5, 5, 1, 2, 3))
    assert naive.tzinfo == UTC
    assert naive.isoformat() == "2013-05-05T01:02:03+00:00"
    tz = timezone(timedelta(hours=-7))
    aware = arrow.get(datetime(2013, 5, 5, tzinfo=tz))
    assert aware.isoformat() == "2013-05-05T00:00:00-07:00"


@pytest.mark.parametrize(
    "value, expected",
    [
        (date(2013, 5, 5), "2013-05-05T00:00:00+00:00"),
        (time.gmtime(0), "1970-01-01T00:00:00+00:00"),
        ((2013, 18, 7), "2013-05-05T00:00:00+00:00"),
    ],
)
def test_get_date_struct_time_and_iso_calendar_tuple(value, expected):
    assert arrow.get(value).isoformat() == expected


def test_get_arrow_returns_equal_copy():
    original = Arrow(2020, 1, 2, 3, 4, 5, tzinfo="+01:00")
    copy = arrow.get(original)
    assert copy == original
    assert copy is not original
    assert copy.utcoffset() == original.utcoffset()


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (datetime(2013, 5, 5), "+09:00", "2013-05-05T00:00:00+09:00"),
        (date(2013, 5, 5), timezone(timedelta(hours=2)), "2013-05-05T00:00:00+02:00"),
    ],
)
def test_get_datetime_or_date_with_replacement_timezone(first, second, expected):
    assert arrow.get(first, second).isoformat() == expected


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss"), {}, Arrow(2013, 5, 5, 12, 30, 45)),
        (
            ("2013-05-05 12:30:45", ["MM/DD/YYYY", "YYYY-MM-DD HH:mm:ss"]),
            {},
            Arrow(2013, 5, 5, 12, 30, 45),
        ),
        (("15 janvier 2020", "D MMMM YYYY"), {"locale": "fr"}, Arrow(2020, 1, 15)),
    ],
)
def test_get_string_with_format_list_or_locale(args, kwargs, expected):
    assert arrow.get(*args, **kwargs) == expected


def test_get_positional_and_keyword_components_build_arrow():
    assert arrow.get(2013, 5, 5) == Arrow(2013, 5, 5)
    assert arrow.get(2013, 5, 5, 12, 30, 45).isoformat() == "2013-05-05T12:30:45+00:00"
    assert arrow.get(year=2013, month=5, day=5, hour=7) == Arrow(2013, 5, 5, 7)


@pytest.mark.parametrize(
    "args",
    [
        (None,),
        (object(),),
        (datetime(2013, 5, 5), 12),
        (1.5, 2.5),
    ],
)
def test_get_unsupported_arguments_raise_type_error(args):
    with pytest.raises(TypeError):
        arrow.get(*args)


@pytest.mark.parametrize("bad", ["this is not a date", (2013, 54, 1)])
def test_get_invalid_values_raise_value_error(bad):
    with pytest.raises(ValueError):
        arrow.get(bad)


@freeze_time("2024-03-05 10:20:30")
def test_factory_with_custom_subclass_returns_subclass_instances():
    class MyArrow(Arrow):
        pass

    custom = arrow.factory(MyArrow)
    assert isinstance(custom, ArrowFactory)
    assert type(custom.get(2013, 5, 5)) is MyArrow
    assert type(custom.get("2013-05-05")) is MyArrow
    assert type(custom.get(1367992474)) is MyArrow
    assert type(custom.utcnow()) is MyArrow
    assert repr(custom.utcnow()) == "<MyArrow [2024-03-05T10:20:30+00:00]>"
