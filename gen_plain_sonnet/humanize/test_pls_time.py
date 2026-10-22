"""Unit tests for humanize.time functions."""

import datetime as dt

import pytest
from freezegun import freeze_time

import humanize
from humanize import (
    naturaldate,
    naturalday,
    naturaldelta,
    naturaltime,
    precisedelta,
)
from humanize.time import (
    Unit,
    _quotient_and_remainder,
    _rounding_by_fmt,
    _suitable_minimum_unit,
    _suppress_lower_units,
)


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


# naturaldelta ----------------------------------------------------------


@pytest.mark.parametrize(
    "delta, expected",
    [
        (dt.timedelta(seconds=0), "a moment"),
        (dt.timedelta(seconds=1), "a second"),
        (dt.timedelta(seconds=30), "30 seconds"),
        (dt.timedelta(seconds=59), "59 seconds"),
        (dt.timedelta(seconds=60), "a minute"),
        (dt.timedelta(seconds=120), "2 minutes"),
        (dt.timedelta(seconds=3599), "an hour"),
        (dt.timedelta(seconds=3600), "an hour"),
        (dt.timedelta(hours=2), "2 hours"),
        (dt.timedelta(hours=23), "23 hours"),
        (dt.timedelta(seconds=86399), "a day"),
        (dt.timedelta(days=1), "a day"),
        (dt.timedelta(days=2), "2 days"),
        (dt.timedelta(days=30), "a month"),
        (dt.timedelta(days=60), "2 months"),
        (dt.timedelta(days=365), "a year"),
        (dt.timedelta(days=366), "1 year, 1 day"),
        (dt.timedelta(days=370), "1 year, 5 days"),
        (dt.timedelta(days=400), "1 year, 1 month"),
        (dt.timedelta(days=730), "2 years"),
        (dt.timedelta(days=365 * 1500), "1,500 years"),
        (dt.timedelta(seconds=-30), "30 seconds"),
    ],
)
def test_naturaldelta_timedeltas(delta, expected):
    assert naturaldelta(delta) == expected


def test_naturaldelta_accepts_numbers_of_seconds():
    assert naturaldelta(30) == "30 seconds"
    assert naturaldelta(30.4) == "30 seconds"
    assert naturaldelta("30") == "30 seconds"
    assert naturaldelta(7200) == "2 hours"
    assert naturaldelta(0.5) == "a moment"


def test_naturaldelta_months_false_uses_days():
    assert naturaldelta(dt.timedelta(days=30), months=False) == "30 days"
    assert naturaldelta(dt.timedelta(days=40), months=False) == "40 days"
    assert naturaldelta(dt.timedelta(days=400), months=False) == "1 year, 35 days"


def test_naturaldelta_minimum_unit_milliseconds_and_microseconds():
    assert naturaldelta(dt.timedelta(milliseconds=500), minimum_unit="milliseconds") == (
        "500 milliseconds"
    )
    assert naturaldelta(dt.timedelta(microseconds=5), minimum_unit="microseconds") == (
        "5 microseconds"
    )
    assert naturaldelta(dt.timedelta(microseconds=1), minimum_unit="microseconds") == (
        "1 microsecond"
    )
    assert naturaldelta(dt.timedelta(milliseconds=2), minimum_unit="microseconds") == (
        "2 milliseconds"
    )
    assert naturaldelta(dt.timedelta(milliseconds=500)) == "a moment"


def test_naturaldelta_invalid_minimum_unit_raises():
    with pytest.raises(ValueError):
        naturaldelta(dt.timedelta(hours=1), minimum_unit="hours")
    with pytest.raises(KeyError):
        naturaldelta(dt.timedelta(hours=1), minimum_unit="bogus")


def test_naturaldelta_non_numeric_values_returned_as_str():
    assert naturaldelta("abc") == "abc"
    assert naturaldelta(None) == "None"
    assert naturaldelta("NaN") == "NaN"
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("nan")) == "nan"


def test_naturaldelta_too_large_finite_value_raises_overflow():
    with pytest.raises(OverflowError):
        naturaldelta(1e20)


# naturaltime -----------------------------------------------------------

NOW = dt.datetime(2020, 1, 1, 12, 0, 0)


def test_naturaltime_numbers_default_to_past_and_future_flag():
    assert naturaltime(30) == "30 seconds ago"
    assert naturaltime(30, future=True) == "30 seconds from now"
    assert naturaltime(120) == "2 minutes ago"
    assert naturaltime(3 * 3600) == "3 hours ago"
    assert naturaltime(3 * 86400) == "3 days ago"


def test_naturaltime_zero_is_now():
    assert naturaltime(0) == "now"
    assert naturaltime(dt.timedelta(0)) == "now"
    assert naturaltime(NOW, when=NOW) == "now"


def test_naturaltime_datetimes_relative_to_when():
    assert naturaltime(NOW - dt.timedelta(seconds=30), when=NOW) == "30 seconds ago"
    assert naturaltime(NOW - dt.timedelta(days=2), when=NOW) == "2 days ago"
    assert naturaltime(NOW - dt.timedelta(days=400), when=NOW) == "1 year, 1 month ago"
    assert naturaltime(NOW + dt.timedelta(minutes=1), when=NOW) == "a minute from now"
    assert naturaltime(NOW + dt.timedelta(hours=5), when=NOW) == "5 hours from now"


def test_naturaltime_timedeltas_sign_determines_tense():
    assert naturaltime(dt.timedelta(seconds=30), when=NOW) == "30 seconds ago"
    assert naturaltime(dt.timedelta(seconds=-30), when=NOW) == "30 seconds from now"


@freeze_time("2024-03-15 12:00:00")
def test_naturaltime_uses_current_time_when_not_given():
    assert naturaltime(dt.datetime(2024, 3, 15, 11, 59, 30)) == "30 seconds ago"
    assert naturaltime(dt.datetime(2024, 3, 15, 14, 0, 0)) == "2 hours from now"
    assert naturaltime(dt.datetime(2024, 3, 13, 12, 0, 0)) == "2 days ago"


def test_naturaltime_months_and_minimum_unit_passthrough():
    past = NOW - dt.timedelta(days=400)
    assert naturaltime(past, months=False, when=NOW) == "1 year, 35 days ago"
    small = NOW - dt.timedelta(milliseconds=500)
    assert naturaltime(small, minimum_unit="milliseconds", when=NOW) == (
        "500 milliseconds ago"
    )


def test_naturaltime_non_numeric_returns_str():
    assert naturaltime("abc") == "abc"
    assert naturaltime(None) == "None"


# naturalday / naturaldate ----------------------------------------------


@freeze_time("2024-03-15 10:30:00")
def test_naturalday_today_tomorrow_yesterday():
    assert naturalday(dt.date(2024, 3, 15)) == "today"
    assert naturalday(dt.date(2024, 3, 16)) == "tomorrow"
    assert naturalday(dt.date(2024, 3, 14)) == "yesterday"
    assert naturalday(dt.datetime(2024, 3, 15, 23, 59)) == "today"


@freeze_time("2024-03-15 10:30:00")
def test_naturalday_other_days_use_format():
    assert naturalday(dt.date(2024, 3, 20)) == "Mar 20"
    assert naturalday(dt.date(2024, 3, 13)) == "Mar 13"
    assert naturalday(dt.date(2024, 3, 1)) == "Mar 01"
    assert naturalday(dt.date(2024, 3, 20), format="%Y-%m-%d") == "2024-03-20"
    assert naturalday(dt.date(2024, 3, 20), "%d/%m") == "20/03"


@freeze_time("2024-03-15 10:30:00")
def test_naturalday_non_date_values_returned_as_str():
    assert naturalday("abc") == "abc"
    assert naturalday(None) == "None"
    assert naturalday(12) == "12"
    assert naturaldate("abc") == "abc"


@freeze_time("2024-03-15 10:30:00")
def test_naturaldate_near_dates_omit_year_far_dates_include_it():
    assert naturaldate(dt.date(2024, 3, 15)) == "today"
    assert naturaldate(dt.date(2024, 3, 16)) == "tomorrow"
    assert naturaldate(dt.date(2024, 3, 14)) == "yesterday"
    assert naturaldate(dt.date(2024, 6, 1)) == "Jun 01"
    assert naturaldate(dt.date(2024, 8, 14)) == "Aug 14"
    assert naturaldate(dt.date(2024, 8, 15)) == "Aug 15 2024"
    assert naturaldate(dt.date(2024, 9, 15)) == "Sep 15 2024"
    assert naturaldate(dt.date(2023, 10, 1)) == "Oct 01 2023"
    assert naturaldate(dt.date(2023, 3, 15)) == "Mar 15 2023"


# precisedelta ----------------------------------------------------------


def test_precisedelta_documented_examples():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"
    assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"
    assert precisedelta(delta, minimum_unit="microseconds") == (
        "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )
    assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_minimum_unit_and_suppress_documented_examples():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert precisedelta(
        delta, suppress=["seconds", "milliseconds", "microseconds"]
    ) == "1.50 minutes"
    assert precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == "0.02 minutes"
    assert precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == "0 minutes"


def test_precisedelta_simple_values():
    assert precisedelta(0) == "0 seconds"
    assert precisedelta(dt.timedelta(seconds=1)) == "1 second"
    assert precisedelta(60) == "1 minute"
    assert precisedelta(1.5) == "1.50 seconds"
    assert precisedelta(dt.timedelta(days=1, hours=2)) == "1 day and 2 hours"
    assert precisedelta(dt.timedelta(days=2, hours=3, minutes=4, seconds=5)) == (
        "2 days, 3 hours, 4 minutes and 5 seconds"
    )


def test_precisedelta_years_and_months():
    assert precisedelta(dt.timedelta(days=365)) == "1 year"
    assert precisedelta(dt.timedelta(days=400)) == "1 year, 1 month and 4 days"
    assert precisedelta(dt.timedelta(days=400), suppress=["years", "months"]) == "400 days"


def test_precisedelta_minimum_unit_rounds_last_unit():
    delta = dt.timedelta(hours=1, minutes=30, seconds=45)
    assert precisedelta(delta, minimum_unit="minutes") == "1 hour and 30.75 minutes"
    assert precisedelta(
        dt.timedelta(milliseconds=1500), minimum_unit="milliseconds"
    ) == "1 second and 500 milliseconds"


def test_precisedelta_invalid_values_and_units():
    assert precisedelta(None) == "None"
    assert precisedelta("abc") == "abc"
    with pytest.raises(ValueError):
        precisedelta(5, minimum_unit="years", suppress=["years"])
    with pytest.raises(KeyError):
        precisedelta(5, minimum_unit="fortnights")


# internal helpers ------------------------------------------------------


def test_unit_ordering():
    assert Unit.SECONDS < Unit.MINUTES
    assert Unit.YEARS > Unit.MONTHS
    assert Unit.DAYS >= Unit.DAYS
    assert sorted([Unit.DAYS, Unit.MICROSECONDS, Unit.YEARS]) == [
        Unit.MICROSECONDS,
        Unit.DAYS,
        Unit.YEARS,
    ]


def test_quotient_and_remainder_modes():
    assert _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f") == (1.5, 0)
    assert _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f") == (
        0,
        36,
    )
    assert _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f") == (1, 12)


def test_suitable_minimum_unit_and_suppress_lower_units():
    assert _suitable_minimum_unit(Unit.HOURS, []) == Unit.HOURS
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]) == Unit.DAYS
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS]) == Unit.MONTHS
    assert _suppress_lower_units(Unit.SECONDS, [Unit.DAYS]) == {
        Unit.MICROSECONDS,
        Unit.MILLISECONDS,
        Unit.DAYS,
    }


def test_rounding_by_fmt_returns_int_or_float():
    assert _rounding_by_fmt("%d", 3.7) == 3
    assert isinstance(_rounding_by_fmt("%d", 3.7), int)
    assert _rounding_by_fmt("%.2f", 3.14159) == 3.14
    assert isinstance(_rounding_by_fmt("%.2f", 3.14159), float)
