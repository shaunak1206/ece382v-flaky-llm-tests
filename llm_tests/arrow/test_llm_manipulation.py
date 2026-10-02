"""Unit tests for shift, replace, to, span/floor/ceil, range, span_range, interval."""
from datetime import datetime, timedelta, timezone

import pytest
from dateutil import tz

import arrow
from arrow import Arrow


@pytest.fixture
def base():
    return arrow.Arrow(2013, 5, 5, 12, 30, 45, 123456)


# ---- shift ----

@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"years": 1}, "2014-05-05T12:30:45.123456+00:00"),
        ({"months": -1}, "2013-04-05T12:30:45.123456+00:00"),
        ({"weeks": 2}, "2013-05-19T12:30:45.123456+00:00"),
        ({"days": 30}, "2013-06-04T12:30:45.123456+00:00"),
        ({"hours": -13}, "2013-05-04T23:30:45.123456+00:00"),
        ({"minutes": 45}, "2013-05-05T13:15:45.123456+00:00"),
        ({"seconds": 20}, "2013-05-05T12:31:05.123456+00:00"),
        ({"quarters": 1}, "2013-08-05T12:30:45.123456+00:00"),
        ({"years": 1, "months": -1}, "2014-04-05T12:30:45.123456+00:00"),
    ],
)
def test_shift_units(base, kwargs, expected):
    assert base.shift(**kwargs).isoformat() == expected


def test_shift_invalid_unit_raises(base):
    with pytest.raises(ValueError):
        base.shift(fortnights=1)


def test_shift_across_dst_gap_resolves_imaginary_time():
    # 2021-03-14 02:30 does not exist in US/Pacific.
    before = arrow.Arrow(2021, 3, 13, 2, 30, tzinfo="US/Pacific")
    shifted = before.shift(days=1)
    assert shifted.hour == 3
    assert shifted.utcoffset() == timedelta(hours=-7)


# ---- replace ----

def test_replace_absolute_values(base):
    result = base.replace(year=2020, month=2, day=29, hour=0)
    assert result.isoformat() == "2020-02-29T00:30:45.123456+00:00"


def test_replace_tzinfo_changes_offset_not_wall_time(base):
    result = base.replace(tzinfo="US/Pacific")
    assert (result.hour, result.minute) == (12, 30)
    assert result.utcoffset() == timedelta(hours=-7)


# ---- to ----

def test_to_named_timezone(base):
    result = base.to("US/Pacific")
    assert result.isoformat() == "2013-05-05T05:30:45.123456-07:00"
    assert result == base


@pytest.mark.parametrize(
    "tzexpr, hour, offset_hours",
    [
        ("Asia/Tokyo", 21, 9),
        ("Europe/London", 13, 1),
        ("-03:00", 9, -3),
        ("+05:30", 18, 5.5),
        ("utc", 12, 0),
        ("UTC", 12, 0),
    ],
)
def test_to_various_expressions(tzexpr, hour, offset_hours):
    a = arrow.Arrow(2013, 5, 5, 12, 30).to(tzexpr)
    assert a.hour == hour
    assert a.utcoffset() == timedelta(hours=offset_hours)


def test_to_local_roundtrip():
    original = arrow.Arrow(2013, 5, 5, 12)
    assert original.to("local").to("utc") == original


# ---- span / floor / ceil ----

@pytest.mark.parametrize(
    "frame, start, end",
    [
        ("year", "2013-01-01T00:00:00+00:00", "2013-12-31T23:59:59.999999+00:00"),
        ("quarter", "2013-04-01T00:00:00+00:00", "2013-06-30T23:59:59.999999+00:00"),
        ("month", "2013-05-01T00:00:00+00:00", "2013-05-31T23:59:59.999999+00:00"),
        ("week", "2013-04-29T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("day", "2013-05-05T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("hour", "2013-05-05T12:00:00+00:00", "2013-05-05T12:59:59.999999+00:00"),
        ("minute", "2013-05-05T12:30:00+00:00", "2013-05-05T12:30:59.999999+00:00"),
        ("second", "2013-05-05T12:30:45+00:00", "2013-05-05T12:30:45.999999+00:00"),
    ],
)
def test_span_frames(base, frame, start, end):
    floor, ceil = base.span(frame)
    assert floor.isoformat() == start
    assert ceil.isoformat() == end


def test_span_count(base):
    floor, ceil = base.span("day", count=2)
    assert floor.isoformat() == "2013-05-05T00:00:00+00:00"
    assert ceil.isoformat() == "2013-05-06T23:59:59.999999+00:00"


@pytest.mark.parametrize(
    "bounds, start_micro, end_iso",
    [
        ("[)", 0, "2013-05-05T23:59:59.999999+00:00"),
        ("[]", 0, "2013-05-06T00:00:00+00:00"),
        ("()", 1, "2013-05-05T23:59:59.999999+00:00"),
        ("(]", 1, "2013-05-06T00:00:00+00:00"),
    ],
)
def test_span_bounds(base, bounds, start_micro, end_iso):
    floor, ceil = base.span("day", bounds=bounds)
    assert floor.microsecond == start_micro
    assert ceil.isoformat() == end_iso


def test_span_invalid_bounds_raises(base):
    with pytest.raises(ValueError):
        base.span("day", bounds="<>")


def test_span_week_start_sunday():
    wed = arrow.Arrow(2021, 2, 24, 10)
    floor, ceil = wed.span("week", week_start=7)
    assert floor.date().isoformat() == "2021-02-21"
    assert ceil.date().isoformat() == "2021-02-27"


def test_span_exact_starts_at_self(base):
    floor, ceil = base.span("hour", exact=True)
    assert floor == base
    assert ceil == base.shift(hours=1).shift(microseconds=-1)


def test_floor_and_ceil_match_span(base):
    assert base.floor("month") == base.span("month")[0]
    assert base.ceil("month") == base.span("month")[1]
    assert base.floor("hour").isoformat() == "2013-05-05T12:00:00+00:00"
    assert base.ceil("year").isoformat() == "2013-12-31T23:59:59.999999+00:00"


# ---- range / span_range / interval ----

def test_range_hours_exclusive_end_when_not_aligned():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 17, 15)
    result = list(Arrow.range("hour", start, end))
    assert [r.hour for r in result] == [12, 13, 14, 15, 16]
    assert all(r.minute == 30 for r in result)


def test_range_includes_end_when_aligned():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 13, 30)
    assert len(list(Arrow.range("hour", start, end))) == 2


def test_range_with_limit_only():
    result = list(Arrow.range("day", datetime(2013, 5, 5), limit=3))
    assert [r.day for r in result] == [5, 6, 7]


def test_range_months_clips_then_recovers_day():
    result = list(Arrow.range("month", datetime(2013, 1, 31), datetime(2013, 5, 31)))
    assert [r.day for r in result] == [31, 28, 31, 30, 31]


def test_range_with_tz_replaces_timezone():
    result = list(Arrow.range("day", datetime(2013, 5, 5), datetime(2013, 5, 7), tz="US/Pacific"))
    assert len(result) == 3
    assert all(r.utcoffset() == timedelta(hours=-7) for r in result)


def test_span_range_hours():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 15, 15)
    spans = list(Arrow.span_range("hour", start, end))
    assert len(spans) == 4
    assert spans[0][0].isoformat() == "2013-05-05T12:00:00+00:00"
    assert spans[0][1].isoformat() == "2013-05-05T12:59:59.999999+00:00"
    assert spans[-1][0].hour == 15


def test_interval_groups_frames():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 17, 15)
    result = list(Arrow.interval("hour", start, end, 2))
    assert len(result) == 3
    assert result[0][0].isoformat() == "2013-05-05T12:00:00+00:00"
    assert result[0][1].isoformat() == "2013-05-05T13:59:59.999999+00:00"
    assert result[1][0].hour == 14


