"""Unit tests for humanize/dehumanize, comparisons, arithmetic and calendar helpers."""
from datetime import date, datetime, time, timedelta, timezone

import pytest

import arrow
from arrow import Arrow


# ---- humanize relative to the current time ----

def test_humanize_just_now():
    assert arrow.utcnow().humanize() == "just now"


@pytest.mark.parametrize(
    "delta, expected",
    [
        ({"seconds": -30}, "30 seconds ago"),
        ({"minutes": -1}, "a minute ago"),
        ({"minutes": -5}, "5 minutes ago"),
        ({"hours": -1}, "an hour ago"),
        ({"hours": -2}, "2 hours ago"),
        ({"days": -1}, "a day ago"),
        ({"days": -3}, "3 days ago"),
        ({"weeks": -2}, "2 weeks ago"),
        ({"months": -3}, "3 months ago"),
        ({"years": -1}, "a year ago"),
        ({"years": -5}, "5 years ago"),
    ],
)
def test_humanize_past_relative_to_now(delta, expected):
    assert arrow.utcnow().shift(**delta).humanize() == expected


@pytest.mark.parametrize(
    "delta, expected",
    [
        ({"hours": 2, "minutes": 1}, "in 2 hours"),
        ({"days": 3, "hours": 1}, "in 3 days"),
        ({"years": 2, "days": 1}, "in 2 years"),
    ],
)
def test_humanize_future_relative_to_now(delta, expected):
    assert arrow.utcnow().shift(**delta).humanize() == expected


def test_humanize_relative_to_other_arrow():
    earlier = arrow.get(2013, 5, 5, 12)
    later = earlier.shift(hours=4)
    assert later.humanize(earlier) == "in 4 hours"
    assert earlier.humanize(later) == "4 hours ago"


def test_humanize_only_distance():
    a = arrow.get(2013, 5, 5)
    assert a.humanize(arrow.get(2013, 5, 7), only_distance=True) == "2 days"


@pytest.mark.parametrize(
    "granularity, expected",
    [
        ("hour", "in 26 hours"),
        ("day", "in a day"),
        (["day", "hour"], "in a day and 2 hours"),
    ],
)
def test_humanize_granularity(granularity, expected):
    base = arrow.get(2013, 5, 5)
    later = base.shift(hours=26)
    assert later.humanize(base, granularity=granularity) == expected


def test_humanize_invalid_other_type_raises():
    with pytest.raises(TypeError):
        arrow.utcnow().humanize("yesterday")


def test_humanize_french():
    a = arrow.get(2013, 5, 5)
    assert a.humanize(a.shift(days=2), locale="fr") == "il y a 2 jours"
    assert a.shift(hours=3).humanize(a, locale="fr") == "dans 3 heures"


def test_humanize_spanish_and_german():
    now = arrow.get(2013, 5, 5, 12)
    assert now.shift(hours=-3).humanize(now, locale="es") == "hace 3 horas"
    assert now.shift(days=-3).humanize(now, locale="de") == "vor 3 Tagen"


# ---- dehumanize ----

@pytest.mark.parametrize(
    "text, kwargs",
    [
        ("in 2 days", {"days": 2}),
        ("2 days ago", {"days": -2}),
        ("in a month", {"months": 1}),
        ("an hour ago", {"hours": -1}),
        ("in 3 weeks", {"weeks": 3}),
        ("just now", {}),
    ],
)
def test_dehumanize_english(text, kwargs):
    base = arrow.get(2021, 4, 20, 22, 27, 34)
    assert base.dehumanize(text) == base.shift(**kwargs)


def test_dehumanize_roundtrips_humanize():
    base = arrow.get(2021, 4, 20, 22, 0)
    other = base.shift(hours=-5)
    assert base.dehumanize(other.humanize(base)) == other


def test_dehumanize_invalid_string_raises():
    with pytest.raises(ValueError):
        arrow.utcnow().dehumanize("gibberish")


# ---- comparisons and arithmetic ----

def test_equality_across_timezones():
    utc = arrow.get(2013, 5, 5, 12)
    pacific = utc.to("US/Pacific")
    assert utc == pacific
    assert hash(utc) == hash(pacific)


def test_ordering_operators():
    a = arrow.get(2013, 5, 5)
    b = arrow.get(2013, 5, 6)
    assert a < b <= b
    assert b > a >= a
    assert a != b


def test_add_and_subtract_timedelta():
    a = arrow.get(2013, 5, 5, 12)
    assert (a + timedelta(days=1, hours=2)).isoformat() == "2013-05-06T14:00:00+00:00"
    assert (a - timedelta(hours=12)).isoformat() == "2013-05-05T00:00:00+00:00"
    assert (timedelta(days=1) + a).day == 6


def test_subtract_arrows_gives_timedelta():
    a = arrow.get(2013, 5, 5)
    b = arrow.get(2013, 5, 3, 12)
    assert a - b == timedelta(days=1, hours=12)
    assert a - datetime(2013, 5, 4, tzinfo=timezone.utc) == timedelta(days=1)


def test_is_between_default_exclusive():
    start, end = arrow.get(2013, 5, 5), arrow.get(2013, 5, 8)
    assert arrow.get(2013, 5, 6).is_between(start, end)
    assert not start.is_between(start, end)


@pytest.mark.parametrize(
    "bounds, at_start, at_end",
    [("[]", True, True), ("[)", True, False), ("(]", False, True), ("()", False, False)],
)
def test_is_between_bounds(bounds, at_start, at_end):
    start, end = arrow.get(2013, 5, 5), arrow.get(2013, 5, 8)
    assert start.is_between(start, end, bounds) is at_start
    assert end.is_between(start, end, bounds) is at_end


# ---- weekday / calendar helpers ----

def test_weekday_and_isoweekday():
    sunday = arrow.get(2013, 5, 5)
    assert sunday.weekday() == 6
    assert sunday.isoweekday() == 7
    assert arrow.get(2013, 5, 6).weekday() == 0


def test_isocalendar():
    assert arrow.get(2013, 5, 5).isocalendar() == (2013, 18, 7)
    assert tuple(arrow.get(2021, 1, 1).isocalendar())[:2] == (2020, 53)


def test_date_time_and_toordinal_helpers():
    a = arrow.get(2013, 5, 5, 12, 30, 45, tzinfo="Asia/Tokyo")
    assert a.date() == date(2013, 5, 5)
    assert a.time() == time(12, 30, 45)
    assert a.timetz().replace(tzinfo=None) == time(12, 30, 45)
    assert a.toordinal() == date(2013, 5, 5).toordinal()


def test_attribute_access_and_quarter_week():
    a = arrow.get(2013, 8, 15, 6, 7, 8, 9)
    assert (a.year, a.month, a.day, a.hour, a.minute, a.second, a.microsecond) == (
        2013, 8, 15, 6, 7, 8, 9,
    )
    assert a.quarter == 3
    assert a.week == 33


def test_ambiguous_and_imaginary():
    ambiguous = arrow.Arrow(2021, 11, 7, 1, 30, tzinfo="US/Pacific")
    imaginary = arrow.Arrow(2021, 3, 14, 2, 30, tzinfo="US/Pacific")
    assert ambiguous.ambiguous is True
    assert imaginary.imaginary is True
    assert arrow.Arrow(2021, 6, 1, 12, tzinfo="US/Pacific").ambiguous is False


