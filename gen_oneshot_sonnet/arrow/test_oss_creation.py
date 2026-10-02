"""Tests for creating Arrow objects: arrow.get, arrow.now, arrow.utcnow, constructors."""

import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

import arrow

PACIFIC_OFFSETS = (timedelta(hours=-7), timedelta(hours=-8))


# ---------------------------------------------------------------- arrow.get


def test_get_no_arguments_returns_current_utc_time():
    before = datetime.now(timezone.utc)
    result = arrow.get()
    after = datetime.now(timezone.utc)
    assert isinstance(result, arrow.Arrow)
    assert result.utcoffset() == timedelta(0)
    assert before <= result.datetime <= after


def test_get_from_datetimes_and_dates():
    assert arrow.get(datetime(2013, 5, 5)).isoformat() == "2013-05-05T00:00:00+00:00"

    aware = datetime(2013, 5, 5, 12, 0, tzinfo=ZoneInfo("US/Pacific"))
    result = arrow.get(aware)
    assert result.isoformat() == "2013-05-05T12:00:00-07:00"
    assert result.datetime == aware

    assert arrow.get(date(2013, 5, 5)).isoformat() == "2013-05-05T00:00:00+00:00"


def test_get_from_timestamps():
    assert arrow.get(1367992474).isoformat() == "2013-05-08T05:54:34+00:00"
    assert arrow.get(0).isoformat() == "1970-01-01T00:00:00+00:00"
    assert arrow.get(-1).isoformat() == "1969-12-31T23:59:59+00:00"
    assert arrow.get(1367992474.5).microsecond == 500000
    assert arrow.get(Decimal("1367992474.5")) == arrow.get(1367992474.5)
    # millisecond and microsecond timestamps are normalized
    assert arrow.get(1367992474000) == arrow.get(1367992474)
    assert arrow.get(1367992474000000) == arrow.get(1367992474)
    # a numeric string needs an explicit format
    assert arrow.get("1367992474", "X") == arrow.get(1367992474)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-05-05", datetime(2013, 5, 5, tzinfo=timezone.utc)),
        ("2013-05", datetime(2013, 5, 1, tzinfo=timezone.utc)),
        ("2013", datetime(2013, 1, 1, tzinfo=timezone.utc)),
        ("2013-05-05T12:30:45", datetime(2013, 5, 5, 12, 30, 45, tzinfo=timezone.utc)),
        ("2013-05-05 12:30", datetime(2013, 5, 5, 12, 30, tzinfo=timezone.utc)),
        (
            "2013-09-29T01:26:43.830580",
            datetime(2013, 9, 29, 1, 26, 43, 830580, tzinfo=timezone.utc),
        ),
        (
            "20160413T133656.456289",
            datetime(2016, 4, 13, 13, 36, 56, 456289, tzinfo=timezone.utc),
        ),
        ("2013-05-05T12:30:45Z", datetime(2013, 5, 5, 12, 30, 45, tzinfo=timezone.utc)),
        (
            "2013-05-05T12:30:45+02:00",
            datetime(2013, 5, 5, 12, 30, 45, tzinfo=timezone(timedelta(hours=2))),
        ),
        (
            "2013-05-05T12:30:45-0530",
            datetime(2013, 5, 5, 12, 30, 45, tzinfo=timezone(timedelta(hours=-5.5))),
        ),
        ("2013-W18-7", datetime(2013, 5, 5, tzinfo=timezone.utc)),
        ("2013-125", datetime(2013, 5, 5, tzinfo=timezone.utc)),
    ],
)
def test_get_from_iso_strings(text, expected):
    result = arrow.get(text)
    assert result.datetime == expected
    assert result.utcoffset() == expected.utcoffset()


def test_get_from_tzinfo_and_from_arrow():
    zone = timezone(timedelta(hours=3))
    result = arrow.get(zone)  # "now" in that zone
    assert result.utcoffset() == timedelta(hours=3)
    assert abs((arrow.utcnow() - result).total_seconds()) < 5

    original = arrow.Arrow(2013, 5, 5, 12, 30, 45, 123456, tzinfo="US/Pacific")
    copy = arrow.get(original)
    assert copy == original
    assert copy is not original
    assert copy.isoformat() == original.isoformat()


def test_get_from_struct_time_and_iso_calendar_tuple():
    assert arrow.get(time.gmtime(0)).isoformat() == "1970-01-01T00:00:00+00:00"
    assert arrow.get((2013, 18, 7)).isoformat() == "2013-05-05T00:00:00+00:00"
    assert arrow.get((2012, 1, 1)).isoformat() == "2012-01-02T00:00:00+00:00"


def test_get_two_arguments_with_replacement_timezone():
    pacific = arrow.get(datetime(2013, 5, 5), "US/Pacific")
    assert pacific.isoformat() == "2013-05-05T00:00:00-07:00"
    assert arrow.get(date(2013, 5, 5), "US/Pacific").isoformat() == (
        "2013-05-05T00:00:00-07:00"
    )
    assert arrow.get(datetime(2013, 5, 5), timezone(timedelta(hours=1))).isoformat() == (
        "2013-05-05T00:00:00+01:00"
    )


def test_get_with_format_string_and_format_list():
    assert arrow.get("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss").isoformat() == (
        "2013-05-05T12:30:45+00:00"
    )
    assert arrow.get(
        "2013-05-05 12:30:45 America/Chicago", "YYYY-MM-DD HH:mm:ss ZZZ"
    ).isoformat() == "2013-05-05T12:30:45-05:00"
    formats = ["MM/DD/YYYY", "YYYY-MM-DD HH:mm:ss"]
    assert arrow.get("2013-05-05 12:30:45", formats).isoformat() == (
        "2013-05-05T12:30:45+00:00"
    )
    assert arrow.get("05/06/2013", formats).isoformat() == "2013-05-06T00:00:00+00:00"


def test_get_with_tzinfo_keyword():
    assert arrow.get("2013-05-05T12:30:45", tzinfo="US/Pacific").isoformat() == (
        "2013-05-05T12:30:45-07:00"
    )
    assert arrow.get(1367992474, tzinfo="Asia/Tokyo").isoformat() == (
        "2013-05-08T14:54:34+09:00"
    )
    assert arrow.get("2013-05-05", "YYYY-MM-DD", tzinfo="Asia/Tokyo").isoformat() == (
        "2013-05-05T00:00:00+09:00"
    )
    assert arrow.get(datetime(2013, 5, 5), tzinfo=timezone(timedelta(hours=2))).isoformat() == (
        "2013-05-05T00:00:00+02:00"
    )
    assert arrow.get(tzinfo="US/Pacific").utcoffset() in PACIFIC_OFFSETS
    assert arrow.get(tzinfo=timezone(timedelta(hours=5))).utcoffset() == timedelta(hours=5)


def test_get_with_locale_and_whitespace_normalization():
    assert arrow.get("5 janvier 2020", "D MMMM YYYY", locale="fr").date() == date(2020, 1, 5)
    assert arrow.get("2013-05-05  12:30", normalize_whitespace=True).isoformat() == (
        "2013-05-05T12:30:00+00:00"
    )
    assert arrow.get("2013-05-05\t12:30", normalize_whitespace=True).hour == 12
    with pytest.raises(arrow.ParserError):
        arrow.get("2013-05-05  12:30")


def test_get_three_or_more_arguments_use_constructor():
    assert arrow.get(2013, 5, 5, 12, 30, 45).isoformat() == "2013-05-05T12:30:45+00:00"
    assert arrow.get(2013, 5, 5).isoformat() == "2013-05-05T00:00:00+00:00"
    assert arrow.get(2013, 5, 5, tzinfo="US/Pacific").isoformat() == (
        "2013-05-05T00:00:00-07:00"
    )


@pytest.mark.parametrize(
    "args, error",
    [
        ((None,), TypeError),
        ((object(),), TypeError),
        ((True,), TypeError),
        (([2013, 5, 5],), TypeError),
        (((2013, 5),), TypeError),
        (((2013, 54, 1),), ValueError),
        (("not a date",), ValueError),
        (("",), ValueError),
        ((datetime(2013, 1, 1), 5), TypeError),
        ((date(2013, 1, 1), 5), TypeError),
        (("2013-05-05", 5), TypeError),
        (("2013-05-05", "HH:mm"), ValueError),
    ],
)
def test_get_with_invalid_arguments_raises(args, error):
    with pytest.raises(error):
        arrow.get(*args)


# ---------------------------------------------------------------- now / utcnow


def test_utcnow_is_utc_and_current():
    before = datetime.now(timezone.utc)
    result = arrow.utcnow()
    after = datetime.now(timezone.utc)
    assert result.utcoffset() == timedelta(0)
    assert before <= result.datetime <= after


def test_now_defaults_to_local_timezone():
    local = datetime.now().astimezone()
    result = arrow.now()
    assert result.utcoffset() == local.utcoffset()
    assert abs((result.datetime - local).total_seconds()) < 5
    utc = arrow.utcnow()
    assert abs((arrow.now() - utc).total_seconds()) < 5
    assert utc <= arrow.utcnow()


def test_now_with_timezone_expressions():
    assert arrow.now("US/Pacific").utcoffset() in PACIFIC_OFFSETS
    assert arrow.now("+02:00").utcoffset() == timedelta(hours=2)
    assert arrow.now("utc").tzinfo == timezone.utc
    assert arrow.now(timezone(timedelta(hours=-3))).utcoffset() == timedelta(hours=-3)
    assert arrow.now("local").utcoffset() == datetime.now().astimezone().utcoffset()
    with pytest.raises(ValueError):
        arrow.now("Not/AZone")


# ---------------------------------------------------------------- constructors


def test_constructor_defaults_and_timezone_strings():
    simple = arrow.Arrow(2013, 5, 5)
    assert simple.isoformat() == "2013-05-05T00:00:00+00:00"
    full = arrow.Arrow(2013, 5, 5, 12, 30, 45, 123456, tzinfo="US/Pacific")
    assert full.isoformat() == "2013-05-05T12:30:45.123456-07:00"
    assert arrow.Arrow(2013, 5, 5, tzinfo="+05:30").utcoffset() == timedelta(hours=5.5)
    with pytest.raises(ValueError):
        arrow.Arrow(2013, 13, 1)
    with pytest.raises(ValueError):
        arrow.Arrow(2013, 5, 5, tzinfo="Not/AZone")


def test_alternate_constructors():
    assert arrow.Arrow.utcfromtimestamp(0).isoformat() == "1970-01-01T00:00:00+00:00"
    assert arrow.Arrow.fromtimestamp(0, tzinfo="Asia/Tokyo").isoformat() == (
        "1970-01-01T09:00:00+09:00"
    )
    assert arrow.Arrow.fromordinal(737741).isoformat() == "2020-11-12T00:00:00+00:00"
    assert arrow.Arrow.fromdate(date(2020, 2, 29)).isoformat() == "2020-02-29T00:00:00+00:00"
    assert arrow.Arrow.fromdatetime(datetime(2020, 2, 29, 1, 2, 3)).isoformat() == (
        "2020-02-29T01:02:03+00:00"
    )
    assert arrow.Arrow.strptime("20-01-2019 15:49:10", "%d-%m-%Y %H:%M:%S").isoformat() == (
        "2019-01-20T15:49:10+00:00"
    )
    with pytest.raises(ValueError):
        arrow.Arrow.fromtimestamp("not a timestamp")
    with pytest.raises(ValueError):
        arrow.Arrow.utcfromtimestamp(10**30)
    with pytest.raises(ValueError):
        arrow.Arrow.fromordinal(0)
    with pytest.raises(TypeError):
        arrow.Arrow.fromordinal("737741")


def test_min_max_and_custom_factory_type():
    assert arrow.Arrow.min < arrow.Arrow(2000, 1, 1) < arrow.Arrow.max

    class MyArrow(arrow.Arrow):
        pass

    factory = arrow.ArrowFactory(MyArrow)
    assert isinstance(factory.utcnow(), MyArrow)
    assert isinstance(factory.get(2013, 5, 5), MyArrow)
    assert isinstance(factory.get("2013-05-05"), MyArrow)
    assert isinstance(factory.now("UTC"), MyArrow)
