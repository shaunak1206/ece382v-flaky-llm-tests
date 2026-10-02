"""Unit tests for humanize.time: naturaldelta, naturaltime, naturalday,
naturaldate, and precisedelta.

Based only on the public function signatures and docstrings in
subjects/humanize/src/humanize/time.py.
"""

import datetime as dt

import pytest
from freezegun import freeze_time

import humanize
from humanize import time as htime

FROZEN = "2020-06-15 12:00:00"


@pytest.mark.parametrize(
    "delta, expected",
    [
        (dt.timedelta(seconds=1), "a second"),
        (dt.timedelta(seconds=30), "30 seconds"),
        (dt.timedelta(minutes=1), "a minute"),
        (dt.timedelta(minutes=30), "30 minutes"),
        (dt.timedelta(hours=1), "an hour"),
        (dt.timedelta(days=1), "a day"),
        (dt.timedelta(days=500), "1 year, 4 months"),
    ],
)
def test_naturaldelta_timedelta_inputs(delta, expected):
    assert htime.naturaldelta(delta) == expected


def test_naturaldelta_accepts_seconds_as_number():
    assert htime.naturaldelta(500) == "8 minutes"


def test_naturaldelta_minimum_unit_milliseconds():
    assert htime.naturaldelta(0.5, minimum_unit="milliseconds") == "500 milliseconds"


def test_naturaldelta_invalid_minimum_unit_raises():
    with pytest.raises(ValueError):
        htime.naturaldelta(5, minimum_unit="hours")


def test_naturaldelta_non_numeric_string_returned_unchanged():
    assert htime.naturaldelta("not a number") == "not a number"


def test_naturaldelta_nan_and_inf_pass_through():
    assert htime.naturaldelta(float("nan")) == "nan"
    assert htime.naturaldelta(float("inf")) == "inf"


def test_naturaldelta_overflow_error_for_huge_seconds():
    with pytest.raises(OverflowError):
        htime.naturaldelta(10**20)


@freeze_time(FROZEN)
def test_naturaltime_seconds_ago():
    now = dt.datetime.now()
    assert htime.naturaltime(now - dt.timedelta(seconds=5)) == "5 seconds ago"


@freeze_time(FROZEN)
def test_naturaltime_seconds_from_now():
    now = dt.datetime.now()
    assert htime.naturaltime(now + dt.timedelta(seconds=5)) == "5 seconds from now"


@freeze_time(FROZEN)
def test_naturaltime_a_day_ago():
    now = dt.datetime.now()
    assert htime.naturaltime(now - dt.timedelta(days=1)) == "a day ago"


@freeze_time(FROZEN)
def test_naturaltime_moment_becomes_now():
    now = dt.datetime.now()
    assert htime.naturaltime(now) == "now"


@freeze_time(FROZEN)
def test_naturaltime_numeric_seconds_future_flag():
    assert htime.naturaltime(5, future=True) == "5 seconds from now"


def test_naturaltime_respects_when_argument():
    now = dt.datetime(2020, 1, 10, 12, 0, 0)
    when = now - dt.timedelta(days=2)
    value = now - dt.timedelta(days=1)
    assert htime.naturaltime(value, when=when) == "a day from now"


@freeze_time(FROZEN)
def test_naturalday_today():
    assert htime.naturalday(dt.datetime.now()) == "today"


@freeze_time(FROZEN)
def test_naturalday_tomorrow():
    tomorrow = dt.datetime.now() + dt.timedelta(days=1)
    assert htime.naturalday(tomorrow) == "tomorrow"


@freeze_time(FROZEN)
def test_naturalday_yesterday():
    yesterday = dt.datetime.now() - dt.timedelta(days=1)
    assert htime.naturalday(yesterday) == "yesterday"


@freeze_time(FROZEN)
def test_naturalday_far_date_uses_format():
    far = dt.datetime.now() + dt.timedelta(days=10)
    assert htime.naturalday(far) == far.strftime("%b %d")


@freeze_time(FROZEN)
def test_naturaldate_far_date_includes_year():
    far = dt.datetime.now() + dt.timedelta(days=200)
    assert htime.naturaldate(far) == far.strftime("%b %d %Y")


def test_precisedelta_default():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert htime.precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_minimum_unit_microseconds():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert htime.precisedelta(delta, minimum_unit="microseconds") == (
        "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )


def test_precisedelta_suppress_days():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert htime.precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_none_returns_string_none():
    assert htime.precisedelta(None) == "None"


def test_module_level_reexports_match_time_module():
    # humanize's public API re-exports these directly from humanize.time.
    assert humanize.naturaldelta is htime.naturaldelta
    assert humanize.naturaltime is htime.naturaltime
    assert humanize.naturalday is htime.naturalday
    assert humanize.naturaldate is htime.naturaldate
    assert humanize.precisedelta is htime.precisedelta
