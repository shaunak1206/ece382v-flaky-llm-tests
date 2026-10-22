import datetime as dt

from freezegun import freeze_time

import humanize


# naturaldelta

def test_naturaldelta_zero():
    assert humanize.naturaldelta(dt.timedelta(seconds=0)) == "a moment"


def test_naturaldelta_seconds():
    assert humanize.naturaldelta(dt.timedelta(seconds=1)) == "a second"
    assert humanize.naturaldelta(dt.timedelta(seconds=30)) == "30 seconds"


def test_naturaldelta_minutes():
    assert humanize.naturaldelta(dt.timedelta(minutes=1)) == "a minute"
    assert humanize.naturaldelta(dt.timedelta(minutes=5)) == "5 minutes"


def test_naturaldelta_hours():
    assert humanize.naturaldelta(dt.timedelta(hours=1)) == "an hour"
    assert humanize.naturaldelta(dt.timedelta(hours=5)) == "5 hours"


def test_naturaldelta_days():
    assert humanize.naturaldelta(dt.timedelta(days=1)) == "a day"
    assert humanize.naturaldelta(dt.timedelta(days=2)) == "2 days"


def test_naturaldelta_numeric_seconds():
    assert humanize.naturaldelta(7) == "7 seconds"
    assert humanize.naturaldelta(3600) == "an hour"


def test_naturaldelta_invalid_returns_value():
    assert humanize.naturaldelta("foo") == "foo"


# naturaltime

def test_naturaltime_timedelta_is_past():
    assert humanize.naturaltime(dt.timedelta(seconds=30)) == "30 seconds ago"
    assert humanize.naturaltime(dt.timedelta(hours=1)) == "an hour ago"


def test_naturaltime_numbers_and_future_flag():
    assert humanize.naturaltime(30) == "30 seconds ago"
    assert humanize.naturaltime(30, future=True) == "30 seconds from now"


def test_naturaltime_negative_timedelta_is_future():
    assert humanize.naturaltime(-dt.timedelta(minutes=1)) == "a minute from now"


def test_naturaltime_with_when_argument():
    when = dt.datetime(2020, 1, 1, 12, 0, 30)
    past = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert humanize.naturaltime(past, when=when) == "30 seconds ago"
    future = dt.datetime(2020, 1, 1, 12, 2, 30)
    assert humanize.naturaltime(future, when=when) == "2 minutes from now"


@freeze_time("2020-01-01 12:00:00")
def test_naturaltime_datetime_with_frozen_clock():
    assert humanize.naturaltime(dt.datetime(2020, 1, 1, 11, 55, 0)) == "5 minutes ago"
    assert humanize.naturaltime(dt.datetime(2020, 1, 1, 14, 0, 0)) == "2 hours from now"


# naturalday

@freeze_time("2020-06-15 10:00:00")
def test_naturalday_today():
    assert humanize.naturalday(dt.date(2020, 6, 15)) == "today"


@freeze_time("2020-06-15 10:00:00")
def test_naturalday_tomorrow_and_yesterday():
    assert humanize.naturalday(dt.date(2020, 6, 16)) == "tomorrow"
    assert humanize.naturalday(dt.date(2020, 6, 14)) == "yesterday"


@freeze_time("2020-06-15 10:00:00")
def test_naturalday_other_date_default_format():
    assert humanize.naturalday(dt.date(2020, 3, 10)) == "Mar 10"


@freeze_time("2020-06-15 10:00:00")
def test_naturalday_custom_format():
    assert humanize.naturalday(dt.date(2020, 3, 10), format="%Y-%m-%d") == "2020-03-10"


@freeze_time("2020-06-15 10:00:00")
def test_naturalday_accepts_datetime():
    assert humanize.naturalday(dt.datetime(2020, 6, 15, 23, 59)) == "today"


# naturaldate

@freeze_time("2020-06-15 10:00:00")
def test_naturaldate_near_dates_have_no_year():
    assert humanize.naturaldate(dt.date(2020, 6, 15)) == "today"
    assert humanize.naturaldate(dt.date(2020, 6, 5)) == "Jun 05"


@freeze_time("2020-06-15 10:00:00")
def test_naturaldate_far_dates_include_year():
    assert humanize.naturaldate(dt.date(2019, 3, 10)) == "Mar 10 2019"


# precisedelta

def test_precisedelta_doc_example():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert humanize.precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_custom_format():
    delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    assert humanize.precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"


def test_precisedelta_single_units():
    assert humanize.precisedelta(dt.timedelta(seconds=1)) == "1 second"
    assert humanize.precisedelta(dt.timedelta(minutes=1)) == "1 minute"
    assert humanize.precisedelta(dt.timedelta(days=1)) == "1 day"


def test_precisedelta_numeric_seconds():
    assert humanize.precisedelta(3661) == "1 hour, 1 minute and 1 second"


def test_precisedelta_suppress_unit():
    delta = dt.timedelta(days=1, hours=2)
    assert humanize.precisedelta(delta, suppress=["days"]) == "26 hours"
