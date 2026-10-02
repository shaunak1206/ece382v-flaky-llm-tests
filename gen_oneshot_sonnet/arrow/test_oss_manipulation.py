"""Tests for shift, replace, to, span/floor/ceil, range, span_range and interval."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from dateutil.relativedelta import FR, MO

import arrow
from arrow import Arrow

BASE_ARGS = (2013, 5, 5, 12, 30, 45, 123456)  # a Sunday


@pytest.fixture
def base():
    return Arrow(*BASE_ARGS)


def iso(pair):
    return tuple(a.isoformat() for a in pair)


# ---------------------------------------------------------------- shift


def test_shift_relative_units(base):
    assert base.shift(hours=1).isoformat() == "2013-05-05T13:30:45.123456+00:00"
    assert base.shift(years=1, months=-1).isoformat() == "2014-04-05T12:30:45.123456+00:00"
    assert base.shift(days=30).isoformat() == "2013-06-04T12:30:45.123456+00:00"
    assert base.shift(weeks=1).isoformat() == "2013-05-12T12:30:45.123456+00:00"
    assert base.shift(quarters=1).isoformat() == "2013-08-05T12:30:45.123456+00:00"
    assert base.shift(minutes=-31).isoformat() == "2013-05-05T11:59:45.123456+00:00"
    assert base.shift(hours=36).isoformat() == "2013-05-07T00:30:45.123456+00:00"
    assert base.shift(microseconds=876544).isoformat() == "2013-05-05T12:30:46+00:00"
    assert Arrow(2013, 1, 15).shift(months=-2).isoformat() == "2012-11-15T00:00:00+00:00"
    assert base.shift() == base


def test_shift_clips_to_end_of_month_and_rejects_unknown_frames(base):
    assert Arrow(2013, 1, 31).shift(months=1).isoformat() == "2013-02-28T00:00:00+00:00"
    assert Arrow(2012, 1, 31).shift(months=1).isoformat() == "2012-02-29T00:00:00+00:00"
    assert Arrow(2013, 3, 31).shift(months=-1).isoformat() == "2013-02-28T00:00:00+00:00"
    assert Arrow(2012, 2, 29).shift(years=1).isoformat() == "2013-02-28T00:00:00+00:00"
    assert Arrow(2012, 2, 29).shift(years=4).isoformat() == "2016-02-29T00:00:00+00:00"

    before = base.isoformat()
    with pytest.raises(ValueError):
        base.shift(foo=1)
    with pytest.raises(ValueError):
        base.shift(hour=1)  # singular is an absolute replace, not a shift
    shifted = base.shift(days=1)
    assert shifted is not base
    assert base.isoformat() == before


def test_shift_to_weekday(base):
    assert base.shift(weekday=0).isoformat() == "2013-05-06T12:30:45.123456+00:00"
    assert base.shift(weekday=4).isoformat() == "2013-05-10T12:30:45.123456+00:00"
    assert base.shift(weekday=6) == base  # already a Sunday
    assert base.shift(weekday=MO(-1)).isoformat() == "2013-04-29T12:30:45.123456+00:00"
    assert base.shift(weekday=FR(+1)).isoformat() == "2013-05-10T12:30:45.123456+00:00"


def test_shift_resolves_imaginary_time_in_dst_gap():
    before_gap = Arrow(2013, 3, 10, 1, 30, tzinfo="US/Pacific")
    assert before_gap.shift(hours=1).isoformat() == "2013-03-10T03:30:00-07:00"
    # wall-clock arithmetic across the gap keeps the wall time
    assert Arrow(2013, 3, 9, 12, tzinfo="US/Pacific").shift(days=1).isoformat() == (
        "2013-03-10T12:00:00-07:00"
    )


# ---------------------------------------------------------------- replace


def test_replace_values_timezone_and_invalid_arguments(base):
    assert base.replace(year=2014, month=6).isoformat() == "2014-06-05T12:30:45.123456+00:00"
    assert base.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() == (
        "2013-05-05T00:00:00+00:00"
    )
    assert base.replace(day=1).day == 1
    assert base.replace(tzinfo="US/Pacific").isoformat() == "2013-05-05T12:30:45.123456-07:00"
    assert base.replace(tzinfo=timezone(timedelta(hours=2))).utcoffset() == timedelta(hours=2)
    assert base.replace(fold=1).fold == 1
    assert base.isoformat() == "2013-05-05T12:30:45.123456+00:00"

    with pytest.raises(ValueError):
        base.replace(week=1)
    with pytest.raises(ValueError):
        base.replace(quarter=2)
    with pytest.raises(ValueError):
        base.replace(years=1)
    with pytest.raises(ValueError):
        base.replace(foo=1)
    with pytest.raises(ValueError):
        base.replace(tzinfo="Not/AZone")
    with pytest.raises(ValueError):
        Arrow(2013, 5, 31).replace(month=6)


# ---------------------------------------------------------------- to


def test_to_converts_between_timezones():
    utc = Arrow(2013, 5, 9, 3, 49, 12, 311072)
    pacific = utc.to("US/Pacific")
    assert pacific.isoformat() == "2013-05-08T20:49:12.311072-07:00"
    assert utc.to("-07:00").isoformat() == "2013-05-08T20:49:12.311072-07:00"
    assert utc.to("Asia/Kolkata").isoformat() == "2013-05-09T09:19:12.311072+05:30"
    assert utc.to(ZoneInfo("Asia/Tokyo")).isoformat() == "2013-05-09T12:49:12.311072+09:00"
    assert utc.to("utc").tzinfo == timezone.utc
    # same instant, different wall clock
    assert pacific == utc
    assert pacific.timestamp() == utc.timestamp()
    assert utc.to("local").to("utc").isoformat() == utc.isoformat()
    with pytest.raises(ValueError):
        utc.to("Not/AZone")


def test_to_handles_dst_transitions():
    assert Arrow(2013, 1, 10, 12).to("US/Pacific").isoformat() == "2013-01-10T04:00:00-08:00"
    assert Arrow(2013, 3, 10, 12).to("US/Pacific").isoformat() == "2013-03-10T05:00:00-07:00"
    assert Arrow(2013, 3, 10, 9, 59, 59).to("US/Pacific").isoformat() == (
        "2013-03-10T01:59:59-08:00"
    )
    assert Arrow(2013, 3, 10, 10, 0, 0).to("US/Pacific").isoformat() == (
        "2013-03-10T03:00:00-07:00"
    )
    first = Arrow(2013, 11, 3, 8, 30).to("US/Pacific")
    second = Arrow(2013, 11, 3, 9, 30).to("US/Pacific")
    assert first.isoformat() == "2013-11-03T01:30:00-07:00"
    assert second.isoformat() == "2013-11-03T01:30:00-08:00"
    assert first.fold == 0 and second.fold == 1
    assert first.ambiguous and second.ambiguous
    assert Arrow(2013, 3, 10, 2, 30, tzinfo="US/Pacific").imaginary


# ---------------------------------------------------------------- span / floor / ceil


@pytest.mark.parametrize(
    "frame, floor, ceil",
    [
        ("year", "2013-01-01T00:00:00+00:00", "2013-12-31T23:59:59.999999+00:00"),
        ("quarter", "2013-04-01T00:00:00+00:00", "2013-06-30T23:59:59.999999+00:00"),
        ("month", "2013-05-01T00:00:00+00:00", "2013-05-31T23:59:59.999999+00:00"),
        ("week", "2013-04-29T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("day", "2013-05-05T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("days", "2013-05-05T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("hour", "2013-05-05T12:00:00+00:00", "2013-05-05T12:59:59.999999+00:00"),
        ("hours", "2013-05-05T12:00:00+00:00", "2013-05-05T12:59:59.999999+00:00"),
        ("minute", "2013-05-05T12:30:00+00:00", "2013-05-05T12:30:59.999999+00:00"),
        ("second", "2013-05-05T12:30:45+00:00", "2013-05-05T12:30:45.999999+00:00"),
    ],
)
def test_span_frames(base, frame, floor, ceil):
    assert iso(base.span(frame)) == (floor, ceil)


def test_span_week_start_and_quarter_and_count(base):
    assert iso(base.span("week", week_start=7)) == (
        "2013-05-05T00:00:00+00:00",
        "2013-05-11T23:59:59.999999+00:00",
    )
    assert iso(base.span("week", week_start=6)) == (
        "2013-05-04T00:00:00+00:00",
        "2013-05-10T23:59:59.999999+00:00",
    )
    monday = Arrow(2013, 5, 6, 8)
    assert iso(monday.span("week", week_start=7)) == (
        "2013-05-05T00:00:00+00:00",
        "2013-05-11T23:59:59.999999+00:00",
    )
    for offset in range(7):  # every day of Mon May 6 .. Sun May 12 shares a floor
        assert Arrow(2013, 5, 6 + offset, 15).floor("week") == Arrow(2013, 5, 6)

    assert iso(base.span("day", count=2)) == (
        "2013-05-05T00:00:00+00:00",
        "2013-05-06T23:59:59.999999+00:00",
    )
    assert iso(base.span("month", count=3)) == (
        "2013-05-01T00:00:00+00:00",
        "2013-07-31T23:59:59.999999+00:00",
    )
    assert iso(base.span("quarter", count=2)) == (
        "2013-04-01T00:00:00+00:00",
        "2013-09-30T23:59:59.999999+00:00",
    )


def test_span_bounds_and_exact(base):
    assert iso(base.span("day", bounds="[]")) == (
        "2013-05-05T00:00:00+00:00",
        "2013-05-06T00:00:00+00:00",
    )
    assert iso(base.span("day", bounds="()")) == (
        "2013-05-05T00:00:00.000001+00:00",
        "2013-05-05T23:59:59.999999+00:00",
    )
    assert iso(base.span("day", bounds="(]")) == (
        "2013-05-05T00:00:00.000001+00:00",
        "2013-05-06T00:00:00+00:00",
    )
    assert iso(base.span("hour", exact=True)) == (
        "2013-05-05T12:30:45.123456+00:00",
        "2013-05-05T13:30:45.123455+00:00",
    )


def test_floor_ceil_and_invalid_arguments(base):
    assert base.floor("hour") == base.span("hour")[0]
    assert base.ceil("hour") == base.span("hour")[1]
    assert base.floor("week", week_start=7).isoformat() == "2013-05-05T00:00:00+00:00"
    assert base.ceil("week", week_start=7).isoformat() == "2013-05-11T23:59:59.999999+00:00"
    assert base.ceil("minute").isoformat() == "2013-05-05T12:30:59.999999+00:00"
    with pytest.raises(ValueError):
        base.span("fortnight")
    with pytest.raises(ValueError):
        base.span("day", bounds="<>")
    with pytest.raises(ValueError):
        base.span("week", week_start=0)
    with pytest.raises(ValueError):
        base.span("week", week_start=8)


def test_span_over_dst_days_keeps_wall_clock_bounds():
    spring = Arrow(2013, 3, 10, 12, tzinfo="US/Pacific")
    floor, ceil = spring.span("day")
    assert floor.isoformat() == "2013-03-10T00:00:00-08:00"
    assert ceil.isoformat() == "2013-03-10T23:59:59.999999-07:00"
    assert round(ceil.timestamp() - floor.timestamp()) == 23 * 3600

    fall = Arrow(2013, 11, 3, 12, tzinfo="US/Pacific")
    floor, ceil = fall.span("day")
    assert round(ceil.timestamp() - floor.timestamp()) == 25 * 3600


# ---------------------------------------------------------------- range


def test_range_includes_end_and_supports_limit():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 17, 15)
    hours = list(Arrow.range("hour", start, end))
    assert [h.isoformat() for h in hours] == [
        "2013-05-05T12:30:00+00:00",
        "2013-05-05T13:30:00+00:00",
        "2013-05-05T14:30:00+00:00",
        "2013-05-05T15:30:00+00:00",
        "2013-05-05T16:30:00+00:00",
    ]
    assert len(list(Arrow.range("hours", start, end))) == 5
    assert len(list(Arrow.range("hour", start, datetime(2013, 5, 5, 13, 30)))) == 2
    assert len(list(Arrow.range("day", Arrow(2013, 1, 1), Arrow(2013, 1, 5)))) == 5
    assert [d.day for d in Arrow.range("day", start, limit=3)] == [5, 6, 7]
    assert len(list(Arrow.range("hour", start, end, limit=2))) == 2
    assert list(Arrow.range("day", end, start)) == []
    with pytest.raises(ValueError):
        list(Arrow.range("day", start))
    with pytest.raises(ValueError):
        list(Arrow.range("fortnight", start, end))


def test_range_keeps_day_of_month_when_clipped():
    months = list(Arrow.range("month", datetime(2013, 1, 31), datetime(2013, 5, 31)))
    assert [(m.month, m.day) for m in months] == [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31)]

    years = list(Arrow.range("year", datetime(2012, 2, 29), datetime(2016, 3, 1)))
    assert [(y.year, y.day) for y in years] == [
        (2012, 29),
        (2013, 28),
        (2014, 28),
        (2015, 28),
        (2016, 29),
    ]


def test_range_other_frames_and_timezone():
    weeks = Arrow.range("week", datetime(2013, 5, 5), datetime(2013, 5, 26))
    assert [w.day for w in weeks] == [5, 12, 19, 26]

    quarters = Arrow.range("quarter", datetime(2013, 1, 1), datetime(2013, 12, 31))
    assert [q.month for q in quarters] == [1, 4, 7, 10]

    days = list(
        Arrow.range("day", datetime(2013, 1, 1), datetime(2013, 1, 3), tz="US/Pacific")
    )
    assert [d.isoformat() for d in days] == [
        "2013-01-01T00:00:00-08:00",
        "2013-01-02T00:00:00-08:00",
        "2013-01-03T00:00:00-08:00",
    ]

    across_dst = list(
        Arrow.range(
            "day",
            Arrow(2013, 3, 9, 12, tzinfo="US/Pacific"),
            Arrow(2013, 3, 11, 12, tzinfo="US/Pacific"),
        )
    )
    assert [d.hour for d in across_dst] == [12, 12, 12]
    assert [d.utcoffset() for d in across_dst] == [
        timedelta(hours=-8),
        timedelta(hours=-7),
        timedelta(hours=-7),
    ]


# ---------------------------------------------------------------- span_range / interval


def test_span_range_options():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 17, 15)
    spans = list(Arrow.span_range("hour", start, end))
    assert len(spans) == 6
    assert iso(spans[0]) == ("2013-05-05T12:00:00+00:00", "2013-05-05T12:59:59.999999+00:00")
    assert iso(spans[-1]) == ("2013-05-05T17:00:00+00:00", "2013-05-05T17:59:59.999999+00:00")

    days = list(Arrow.span_range("day", datetime(2013, 1, 1, 5), datetime(2013, 1, 3, 5)))
    assert [d[0].day for d in days] == [1, 2, 3]
    assert all(d[1].isoformat().endswith("23:59:59.999999+00:00") for d in days)

    months = list(Arrow.span_range("month", datetime(2013, 1, 15), datetime(2013, 3, 10)))
    assert [m[1].day for m in months] == [31, 28, 31]

    inclusive = list(
        Arrow.span_range("day", datetime(2013, 1, 1), datetime(2013, 1, 2), bounds="[]")
    )
    assert iso(inclusive[0]) == ("2013-01-01T00:00:00+00:00", "2013-01-02T00:00:00+00:00")

    assert len(list(Arrow.span_range("day", datetime(2013, 1, 1), datetime(2013, 2, 1), limit=2))) == 2

    exact = list(Arrow.span_range("hour", start, end, exact=True))
    assert len(exact) == 5
    assert iso(exact[0]) == ("2013-05-05T12:30:00+00:00", "2013-05-05T13:29:59.999999+00:00")
    assert exact[-1][1].isoformat() == "2013-05-05T17:14:59.999999+00:00"


def test_interval_groups_spans():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 17, 15)
    intervals = list(Arrow.interval("hour", start, end, 2))
    assert [iso(i) for i in intervals] == [
        ("2013-05-05T12:00:00+00:00", "2013-05-05T13:59:59.999999+00:00"),
        ("2013-05-05T14:00:00+00:00", "2013-05-05T15:59:59.999999+00:00"),
        ("2013-05-05T16:00:00+00:00", "2013-05-05T17:59:59.999999+00:00"),
    ]
    assert list(Arrow.interval("hour", start, end, 1)) == list(
        Arrow.span_range("hour", start, end)
    )
    # a short final group is still emitted
    short = list(Arrow.interval("day", datetime(2013, 1, 1), datetime(2013, 1, 3), 2))
    assert len(short) == 2
    assert short[1][0].day == 3
    with pytest.raises(ValueError):
        list(Arrow.interval("hour", start, end, 0))
    assert arrow.Arrow is Arrow
