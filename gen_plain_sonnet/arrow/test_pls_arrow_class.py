"""Tests for the Arrow class in arrow.arrow: construction, shifting, spans, ranges,
humanize/dehumanize, comparisons and arithmetic."""

from datetime import date, datetime, time as dt_time, timedelta, timezone

import pytest
from dateutil.relativedelta import relativedelta
from freezegun import freeze_time

from arrow import Arrow

UTC = timezone.utc
NY = "America/New_York"


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_constructor_defaults_and_timezones():
    plain = Arrow(2013, 5, 5)
    assert plain.tzinfo == UTC
    assert plain.isoformat() == "2013-05-05T00:00:00+00:00"
    full = Arrow(2013, 5, 5, 1, 2, 3, 4, "+02:00")
    assert full.isoformat() == "2013-05-05T01:02:03.000004+02:00"
    assert Arrow.min.year == 1
    assert Arrow.max.year == 9999
    for bad in [(2013, 13, 1), (2013, 2, 30)]:
        with pytest.raises(ValueError):
            Arrow(*bad)


def test_timestamp_constructors():
    expected = "2013-05-08T05:54:34+00:00"
    assert Arrow.fromtimestamp(1367992474, tzinfo="UTC").isoformat() == expected
    # millisecond and microsecond timestamps are normalized
    assert Arrow.fromtimestamp(1367992474000, tzinfo="utc").isoformat() == expected
    assert Arrow.fromtimestamp(1367992474000000, tzinfo="utc").isoformat() == expected
    assert Arrow.fromtimestamp("1367992474", tzinfo="utc").isoformat() == expected
    assert Arrow.utcfromtimestamp(1367992474).isoformat() == expected
    assert Arrow.utcfromtimestamp(0).year == 1970
    with pytest.raises(ValueError):
        Arrow.fromtimestamp("abc")
    with pytest.raises(ValueError):
        Arrow.utcfromtimestamp("abc")
    with pytest.raises(ValueError):
        Arrow.utcfromtimestamp(1e30)


def test_from_datetime_date_strptime_and_ordinal():
    naive = Arrow.fromdatetime(datetime(2013, 5, 5, 1, 2, 3))
    assert naive.isoformat() == "2013-05-05T01:02:03+00:00"
    replaced = Arrow.fromdatetime(datetime(2013, 5, 5, 1, 2, 3), "+02:00")
    assert replaced.isoformat() == "2013-05-05T01:02:03+02:00"
    aware = Arrow.fromdatetime(datetime(2013, 5, 5, tzinfo=timezone(timedelta(hours=-4))))
    assert aware.isoformat() == "2013-05-05T00:00:00-04:00"

    assert Arrow.fromdate(date(2013, 5, 5)).isoformat() == "2013-05-05T00:00:00+00:00"
    assert Arrow.fromdate(date(2013, 5, 5), "+01:00").isoformat() == "2013-05-05T00:00:00+01:00"

    parsed = Arrow.strptime("20-01-2019 15:49:10", "%d-%m-%Y %H:%M:%S")
    assert parsed == Arrow(2019, 1, 20, 15, 49, 10)
    shifted = Arrow.strptime("20-01-2019 15:49:10", "%d-%m-%Y %H:%M:%S", tzinfo="+05:00")
    assert shifted.isoformat() == "2019-01-20T15:49:10+05:00"

    assert Arrow.fromordinal(737741) == Arrow(2020, 11, 12)
    with pytest.raises(ValueError):
        Arrow.fromordinal(0)
    with pytest.raises(TypeError):
        Arrow.fromordinal(1.5)
    with pytest.raises(TypeError):
        Arrow.fromordinal(True)


# ---------------------------------------------------------------------------
# replace / shift
# ---------------------------------------------------------------------------


def test_replace_and_clone_do_not_mutate_original():
    original = Arrow(2013, 5, 5, 12, 30, 45, 123456, tzinfo="+02:00")
    assert original.replace(year=2020, month=1).isoformat() == "2020-01-05T12:30:45.123456+02:00"
    cleared = original.replace(hour=0, minute=0, second=0, microsecond=0)
    assert cleared.isoformat() == "2013-05-05T00:00:00+02:00"
    # tzinfo is replaced, not converted
    assert original.replace(tzinfo="+09:00").isoformat() == "2013-05-05T12:30:45.123456+09:00"
    assert original.replace(tzinfo=UTC).hour == 12
    assert original.isoformat() == "2013-05-05T12:30:45.123456+02:00"
    cloned = original.clone()
    assert cloned == original
    assert cloned is not original


@pytest.mark.parametrize(
    "call, kwargs",
    [
        ("replace", {"week": 1}),
        ("replace", {"quarter": 2}),
        ("replace", {"days": 1}),
        ("replace", {"month": 13}),
        ("shift", {"day": 1}),
        ("shift", {"fortnights": 1}),
    ],
)
def test_replace_and_shift_reject_invalid_keywords(call, kwargs):
    with pytest.raises(ValueError):
        getattr(Arrow(2013, 5, 5), call)(**kwargs)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"years": 1}, "2021-01-31T12:00:00+00:00"),
        ({"months": 1}, "2020-02-29T12:00:00+00:00"),
        ({"months": 13}, "2021-02-28T12:00:00+00:00"),
        ({"quarters": 1}, "2020-04-30T12:00:00+00:00"),
        ({"weeks": 2}, "2020-02-14T12:00:00+00:00"),
        ({"days": -31}, "2019-12-31T12:00:00+00:00"),
        ({"hours": 13}, "2020-02-01T01:00:00+00:00"),
        ({"minutes": -30}, "2020-01-31T11:30:00+00:00"),
        ({"seconds": 90}, "2020-01-31T12:01:30+00:00"),
        ({"microseconds": 5}, "2020-01-31T12:00:00.000005+00:00"),
        ({"years": 1, "months": -1, "days": 1}, "2021-01-01T12:00:00+00:00"),
    ],
)
def test_shift_relative_amounts(kwargs, expected):
    assert Arrow(2020, 1, 31, 12).shift(**kwargs).isoformat() == expected


def test_shift_weekday_moves_forward_only():
    saturday = Arrow(2013, 5, 11, 22, 27)
    assert saturday.shift(weekday=5) == saturday
    assert saturday.shift(weekday=0).isoformat() == "2013-05-13T22:27:00+00:00"


def test_shift_resolves_imaginary_times_across_dst_gap():
    before_gap = Arrow(2021, 3, 13, 2, 30, tzinfo=NY)
    assert before_gap.shift(days=1).isoformat() == "2021-03-14T03:30:00-04:00"
    assert Arrow(2021, 3, 14, 1, 30, tzinfo=NY).shift(hours=1).isoformat() == (
        "2021-03-14T03:30:00-04:00"
    )
    unresolved = before_gap.shift(days=1, check_imaginary=False)
    assert unresolved.imaginary


def test_imaginary_ambiguous_and_fold_properties():
    assert Arrow(2021, 3, 14, 2, 30, tzinfo=NY).imaginary
    assert not Arrow(2021, 3, 14, 3, 30, tzinfo=NY).imaginary
    assert Arrow(2021, 11, 7, 1, 30, tzinfo=NY).ambiguous
    assert not Arrow(2021, 11, 7, 3, 30, tzinfo=NY).ambiguous
    first = Arrow(2021, 11, 7, 1, 30, tzinfo=NY)
    second = Arrow(2021, 11, 7, 1, 30, tzinfo=NY, fold=1)
    assert first.fold == 0 and second.fold == 1
    assert first.utcoffset() == timedelta(hours=-4)
    assert second.utcoffset() == timedelta(hours=-5)


def test_to_converts_timezone_without_changing_instant():
    base = Arrow(2013, 5, 5, 12, 30, tzinfo="utc")
    shifted = base.to("+02:00")
    assert shifted.isoformat() == "2013-05-05T14:30:00+02:00"
    assert shifted == base
    assert base.to(timezone(timedelta(hours=-5))).hour == 7
    assert base.to("Asia/Tokyo").hour == 21
    assert Arrow(2013, 5, 5, 23, tzinfo="utc").to("+02:00").day == 6
    assert base.to("+02:00").to("utc").isoformat() == base.isoformat()
    with pytest.raises(ValueError):
        base.to("Not/A_Zone")


# ---------------------------------------------------------------------------
# span / floor / ceil
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame, floor_iso, ceil_iso",
    [
        ("second", "2013-05-05T12:30:45+00:00", "2013-05-05T12:30:45.999999+00:00"),
        ("minute", "2013-05-05T12:30:00+00:00", "2013-05-05T12:30:59.999999+00:00"),
        ("hour", "2013-05-05T12:00:00+00:00", "2013-05-05T12:59:59.999999+00:00"),
        ("day", "2013-05-05T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("days", "2013-05-05T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("week", "2013-04-29T00:00:00+00:00", "2013-05-05T23:59:59.999999+00:00"),
        ("month", "2013-05-01T00:00:00+00:00", "2013-05-31T23:59:59.999999+00:00"),
        ("quarter", "2013-04-01T00:00:00+00:00", "2013-06-30T23:59:59.999999+00:00"),
        ("year", "2013-01-01T00:00:00+00:00", "2013-12-31T23:59:59.999999+00:00"),
    ],
)
def test_span_frames(frame, floor_iso, ceil_iso):
    floor, ceil = Arrow(2013, 5, 5, 12, 30, 45, 123456).span(frame)
    assert floor.isoformat() == floor_iso
    assert ceil.isoformat() == ceil_iso


def test_span_options_bounds_exact_count_and_week_start():
    base = Arrow(2013, 5, 5, 12, 30)
    floor, ceil = base.span("day", bounds="[]")
    assert (floor.isoformat(), ceil.isoformat()) == (
        "2013-05-05T00:00:00+00:00",
        "2013-05-06T00:00:00+00:00",
    )
    floor, ceil = base.span("day", bounds="()")
    assert (floor.isoformat(), ceil.isoformat()) == (
        "2013-05-05T00:00:00.000001+00:00",
        "2013-05-05T23:59:59.999999+00:00",
    )
    floor, ceil = base.span("hour", exact=True)
    assert (floor.isoformat(), ceil.isoformat()) == (
        "2013-05-05T12:30:00+00:00",
        "2013-05-05T13:29:59.999999+00:00",
    )
    assert base.span("day", count=2)[1].isoformat() == "2013-05-06T23:59:59.999999+00:00"
    assert base.floor("week", week_start=7).isoformat() == "2013-05-05T00:00:00+00:00"
    assert base.ceil("week", week_start=7).isoformat() == "2013-05-11T23:59:59.999999+00:00"
    assert base.floor("week", week_start=3).isoformat() == "2013-05-01T00:00:00+00:00"
    assert base.floor("hour") == base.span("hour")[0]
    assert base.ceil("hour") == base.span("hour")[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame": "fortnight"},
        {"frame": "day", "bounds": "]["},
        {"frame": "week", "week_start": 0},
        {"frame": "week", "week_start": 8},
    ],
)
def test_span_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        Arrow(2013, 5, 5).span(**kwargs)


# ---------------------------------------------------------------------------
# range / span_range / interval
# ---------------------------------------------------------------------------


def test_range_includes_end_and_honors_limit_and_tz():
    start = datetime(2013, 5, 5, 12, 30)
    end = datetime(2013, 5, 5, 15, 30)
    hours = list(Arrow.range("hour", start, end))
    assert len(hours) == 4
    assert hours[0] == Arrow(2013, 5, 5, 12, 30)
    assert hours[-1] == Arrow(2013, 5, 5, 15, 30)
    assert list(Arrow.range("hours", start, end)) == hours
    assert len(list(Arrow.range("hour", start, limit=3))) == 3
    assert len(list(Arrow.range("hour", start, end, limit=2))) == 2
    days = list(Arrow.range("day", Arrow(2013, 5, 5), Arrow(2013, 5, 8)))
    assert len(days) == 4
    zoned = list(Arrow.range("hour", datetime(2013, 5, 5), datetime(2013, 5, 5, 2), tz="+02:00"))
    assert [r.isoformat() for r in zoned][0] == "2013-05-05T00:00:00+02:00"
    assert len(zoned) == 3
    quarters = list(Arrow.range("quarter", datetime(2013, 1, 1), datetime(2013, 12, 31)))
    assert [q.month for q in quarters] == [1, 4, 7, 10]


def test_range_month_end_clipping_restores_original_day():
    result = list(Arrow.range("month", datetime(2013, 1, 31), datetime(2013, 4, 30)))
    assert [(r.month, r.day) for r in result] == [(1, 31), (2, 28), (3, 31), (4, 30)]


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (("day", datetime(2013, 5, 5)), {}),
        (("fortnight", datetime(2013, 5, 5), datetime(2013, 5, 9)), {}),
    ],
)
def test_range_invalid_arguments_raise(args, kwargs):
    with pytest.raises(ValueError):
        list(Arrow.range(*args, **kwargs))


def test_span_range_and_interval():
    spans = list(Arrow.span_range("day", datetime(2013, 5, 5, 12), datetime(2013, 5, 7, 12)))
    assert len(spans) == 3
    assert spans[0][0].isoformat() == "2013-05-05T00:00:00+00:00"
    assert spans[0][1].isoformat() == "2013-05-05T23:59:59.999999+00:00"
    assert spans[2][0].day == 7

    start, end = datetime(2013, 5, 5, 12, 30), datetime(2013, 5, 5, 17, 15)
    pairs = list(Arrow.interval("hour", start, end, 2))
    assert len(pairs) == 3
    assert pairs[0][0].isoformat() == "2013-05-05T12:00:00+00:00"
    assert pairs[0][1].isoformat() == "2013-05-05T13:59:59.999999+00:00"
    assert pairs[2][0].hour == 16
    assert len(list(Arrow.interval("hour", start, end, 4))) == 2
    with pytest.raises(ValueError):
        list(Arrow.interval("hour", start, end, 0))


# ---------------------------------------------------------------------------
# humanize / dehumanize
# ---------------------------------------------------------------------------

BASE = Arrow(2013, 5, 5, 12, 0, 0)


@pytest.mark.parametrize(
    "delta, expected",
    [
        ({}, "just now"),
        ({"seconds": 5}, "just now"),
        ({"seconds": 44}, "in 44 seconds"),
        ({"seconds": -44}, "44 seconds ago"),
        ({"seconds": -70}, "a minute ago"),
        ({"minutes": 5}, "in 5 minutes"),
        ({"minutes": -90}, "an hour ago"),
        ({"hours": 3}, "in 3 hours"),
        ({"hours": -30}, "a day ago"),
        ({"days": -3}, "3 days ago"),
        ({"days": 8}, "in a week"),
        ({"days": -14}, "2 weeks ago"),
        ({"months": 1}, "in a month"),
        ({"months": -3}, "3 months ago"),
        ({"years": -1}, "a year ago"),
        ({"years": 5}, "in 5 years"),
    ],
)
def test_humanize_auto_granularity(delta, expected):
    assert BASE.shift(**delta).humanize(BASE) == expected


def test_humanize_options_and_errors():
    earlier = BASE.shift(hours=-2)
    assert earlier.humanize(BASE, only_distance=True) == "2 hours"
    assert BASE.humanize(BASE, only_distance=True) == "instantly"
    assert BASE.shift(hours=-5).humanize(BASE, granularity="hour") == "5 hours ago"
    assert BASE.shift(days=3).humanize(BASE, granularity="day") == "in 3 days"
    assert BASE.shift(seconds=1).humanize(BASE, granularity="second") == "just now"
    assert BASE.shift(seconds=90).humanize(BASE, granularity="second") == "in 90 seconds"
    assert BASE.shift(days=3).humanize(BASE, granularity=["day"]) == "in 3 days"
    past = BASE.shift(hours=-2, minutes=-30)
    assert past.humanize(BASE, granularity=["hour", "minute"]) == "2 hours and 30 minutes ago"
    future = BASE.shift(hours=2, minutes=30)
    assert future.humanize(BASE, granularity=["hour", "minute"]) == "in 2 hours and 30 minutes"
    assert BASE.humanize(datetime(2013, 5, 5, 10, 0, 0)) == "in 2 hours"
    with pytest.raises(ValueError):
        BASE.humanize(BASE, granularity="decade")
    with pytest.raises(ValueError):
        BASE.humanize(BASE, granularity=[])
    with pytest.raises(TypeError):
        BASE.humanize(5)


@freeze_time("2013-05-05 12:00:00")
def test_humanize_without_other_uses_current_time():
    assert Arrow.utcnow().shift(hours=-2).humanize() == "2 hours ago"
    assert Arrow(2013, 5, 5, 14, 0, 0, tzinfo="+02:00").humanize() == "just now"
    assert Arrow.utcnow().shift(days=3).humanize() == "in 3 days"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("in 2 hours", "2013-05-05T14:00:00+00:00"),
        ("3 days ago", "2013-05-02T12:00:00+00:00"),
        ("in a month", "2013-06-05T12:00:00+00:00"),
        ("5 minutes ago", "2013-05-05T11:55:00+00:00"),
        ("2 weeks ago", "2013-04-21T12:00:00+00:00"),
        ("in 2 hours and 30 minutes", "2013-05-05T14:30:00+00:00"),
        ("just now", "2013-05-05T12:00:00+00:00"),
    ],
)
def test_dehumanize_english(text, expected):
    assert BASE.dehumanize(text).isoformat() == expected


@pytest.mark.parametrize("text", ["gibberish", "2 hours"])
def test_dehumanize_invalid_strings_raise(text):
    with pytest.raises(ValueError):
        BASE.dehumanize(text)


# ---------------------------------------------------------------------------
# queries, comparisons, arithmetic
# ---------------------------------------------------------------------------


def test_is_between_bounds():
    start, end = Arrow(2013, 5, 5), Arrow(2013, 5, 8)
    middle = Arrow(2013, 5, 6)
    assert middle.is_between(start, end)
    assert not Arrow(2013, 5, 9).is_between(start, end, "[]")
    assert not end.is_between(start, end)
    assert end.is_between(start, end, "[]")
    assert not end.is_between(start, end, "[)")
    assert end.is_between(start, end, "(]")
    assert start.is_between(start, end, "[)")
    assert not start.is_between(start, end, "()")
    with pytest.raises(ValueError):
        middle.is_between(start, end, "<>")
    with pytest.raises(TypeError):
        middle.is_between(datetime(2013, 5, 5), end)
    with pytest.raises(TypeError):
        middle.is_between(start, datetime(2013, 5, 8))


def test_comparisons_and_hash_across_timezones():
    noon_utc = Arrow(2013, 5, 5, 12, tzinfo="utc")
    same_instant = noon_utc.to("+02:00")
    later = Arrow(2013, 5, 5, 13)
    assert noon_utc == same_instant
    assert noon_utc == same_instant.datetime
    assert noon_utc != later
    assert noon_utc < later and noon_utc <= later
    assert later > noon_utc and later >= noon_utc
    assert noon_utc < datetime(2014, 1, 1, tzinfo=UTC)
    assert noon_utc != "2013-05-05"
    assert not (noon_utc == 5)
    assert hash(noon_utc) == hash(same_instant)
    assert len({noon_utc, same_instant}) == 1
    with pytest.raises(TypeError):
        noon_utc < 5


def test_arithmetic_operators():
    moment = Arrow(2013, 5, 5, 12, tzinfo="+02:00")
    plus = moment + timedelta(hours=3)
    assert isinstance(plus, Arrow)
    assert plus.isoformat() == "2013-05-05T15:00:00+02:00"
    assert (timedelta(hours=3) + moment) == plus
    assert (moment - timedelta(days=1)).isoformat() == "2013-05-04T12:00:00+02:00"
    assert (moment + relativedelta(months=1)).isoformat() == "2013-06-05T12:00:00+02:00"
    assert Arrow(2013, 5, 5, 12) - Arrow(2013, 5, 5, 10) == timedelta(hours=2)
    assert moment - datetime(2013, 5, 5, 0, tzinfo=UTC) == timedelta(hours=10)
    assert datetime(2013, 5, 6, tzinfo=UTC) - Arrow(2013, 5, 5) == timedelta(days=1)
    with pytest.raises(TypeError):
        moment + 1
    with pytest.raises(TypeError):
        moment - "x"


# ---------------------------------------------------------------------------
# attributes and datetime-style methods
# ---------------------------------------------------------------------------


def test_component_attributes_week_quarter_and_timestamps():
    moment = Arrow(2013, 5, 8, 5, 54, 34, 250000)
    assert (moment.year, moment.month, moment.day) == (2013, 5, 8)
    assert (moment.hour, moment.minute, moment.second, moment.microsecond) == (5, 54, 34, 250000)
    assert Arrow(2013, 5, 5).week == 18
    assert [Arrow(2013, m, 1).quarter for m in (1, 3, 4, 6, 7, 9, 10, 12)] == [1, 1, 2, 2, 3, 3, 4, 4]
    whole = Arrow(2013, 5, 8, 5, 54, 34)
    assert whole.int_timestamp == 1367992474
    assert whole.timestamp() == 1367992474.0
    assert whole.float_timestamp == 1367992474.0
    assert moment.timestamp() == 1367992474.25
    with pytest.raises(AttributeError):
        Arrow(2013, 5, 5).nonexistent_attribute


def test_datetime_style_methods_delegate_to_datetime():
    moment = Arrow(2013, 5, 5, 1, 2, 3, 4, tzinfo="+02:00")
    assert moment.datetime == datetime(2013, 5, 5, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=2)))
    assert moment.naive == datetime(2013, 5, 5, 1, 2, 3, 4)
    assert moment.naive.tzinfo is None
    assert moment.date() == date(2013, 5, 5)
    assert moment.time() == dt_time(1, 2, 3, 4)
    assert moment.timetz().utcoffset() == timedelta(hours=2)
    assert moment.utcoffset() == timedelta(hours=2)
    assert moment.weekday() == 6
    assert moment.isoweekday() == 7
    assert tuple(moment.isocalendar()) == (2013, 18, 7)
    assert moment.toordinal() == date(2013, 5, 5).toordinal()
    assert moment.timetuple().tm_yday == 125
    utc_tuple = moment.utctimetuple()
    assert (utc_tuple.tm_mday, utc_tuple.tm_hour) == (4, 23)
    converted = moment.astimezone(timezone(timedelta(hours=-5)))
    assert isinstance(converted, datetime)
    assert (converted.day, converted.hour) == (4, 18)


def test_string_representations():
    moment = Arrow(2013, 5, 5, 12, 30, 45)
    assert str(moment) == "2013-05-05T12:30:45+00:00"
    assert repr(moment) == "<Arrow [2013-05-05T12:30:45+00:00]>"
    assert moment.isoformat() == "2013-05-05T12:30:45+00:00"
    assert moment.isoformat(sep=" ") == "2013-05-05 12:30:45+00:00"
    assert moment.isoformat(timespec="minutes") == "2013-05-05T12:30+00:00"
    assert moment.ctime() == "Sun May  5 12:30:45 2013"
    assert moment.strftime("%d-%m-%Y %H:%M:%S") == "05-05-2013 12:30:45"
    assert moment.for_json() == moment.isoformat()

    class Custom(Arrow):
        pass

    assert repr(Custom(2013, 5, 5)) == "<Custom [2013-05-05T00:00:00+00:00]>"
