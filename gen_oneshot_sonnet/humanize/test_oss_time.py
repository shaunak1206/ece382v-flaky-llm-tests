"""Tests for humanize.time functions."""

import datetime as dt

import pytest
from freezegun import freeze_time

import humanize


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


WHEN = dt.datetime(2024, 6, 15, 12, 0, 0)


# ---------------------------------------------------------------- naturaldelta


@pytest.mark.parametrize(
    "delta, expected",
    [
        (dt.timedelta(0), "a moment"),
        (dt.timedelta(seconds=1), "a second"),
        (dt.timedelta(seconds=30), "30 seconds"),
        (dt.timedelta(seconds=59), "59 seconds"),
        (dt.timedelta(seconds=60), "a minute"),
        (dt.timedelta(seconds=90), "2 minutes"),
        (dt.timedelta(minutes=45), "45 minutes"),
        (dt.timedelta(seconds=3599), "an hour"),
        (dt.timedelta(hours=1), "an hour"),
        (dt.timedelta(hours=2), "2 hours"),
        (dt.timedelta(hours=23), "23 hours"),
        (dt.timedelta(days=1), "a day"),
        (dt.timedelta(days=2), "2 days"),
        (dt.timedelta(days=30), "a month"),
        (dt.timedelta(days=61), "2 months"),
        (dt.timedelta(days=365), "a year"),
        (dt.timedelta(days=366), "1 year, 1 day"),
        (dt.timedelta(days=370), "1 year, 5 days"),
        (dt.timedelta(days=400), "1 year, 1 month"),
        (dt.timedelta(days=730), "2 years"),
        (dt.timedelta(days=365 * 1000), "1,000 years"),
    ],
)
def test_naturaldelta_timedeltas(delta, expected):
    assert humanize.naturaldelta(delta) == expected


def test_naturaldelta_numbers_negatives_and_invalid_values():
    assert humanize.naturaldelta(dt.timedelta(seconds=-60)) == "a minute"
    assert humanize.naturaldelta(dt.timedelta(days=-2)) == "2 days"
    assert humanize.naturaldelta(30) == "30 seconds"
    assert humanize.naturaldelta(30.0) == "30 seconds"
    assert humanize.naturaldelta("30") == "30 seconds"
    assert humanize.naturaldelta(3600) == "an hour"
    assert humanize.naturaldelta("foo") == "foo"
    assert humanize.naturaldelta(None) == "None"
    assert humanize.naturaldelta(float("inf")) == "inf"
    assert humanize.naturaldelta(float("nan")) == "nan"


def test_naturaldelta_months_option():
    assert humanize.naturaldelta(dt.timedelta(days=60), months=False) == "60 days"
    assert (
        humanize.naturaldelta(dt.timedelta(days=400), months=False) == "1 year, 35 days"
    )
    assert humanize.naturaldelta(dt.timedelta(days=60), months=True) == "2 months"


def test_naturaldelta_minimum_unit():
    half_second = dt.timedelta(milliseconds=500)
    assert humanize.naturaldelta(half_second) == "a moment"
    assert humanize.naturaldelta(half_second, minimum_unit="milliseconds") == (
        "500 milliseconds"
    )
    assert humanize.naturaldelta(
        dt.timedelta(microseconds=5), minimum_unit="microseconds"
    ) == "5 microseconds"
    assert humanize.naturaldelta(
        dt.timedelta(microseconds=1), minimum_unit="microseconds"
    ) == "1 microsecond"
    assert humanize.naturaldelta(dt.timedelta(seconds=30), minimum_unit="Seconds") == (
        "30 seconds"
    )


def test_naturaldelta_bad_minimum_unit_and_overflow():
    with pytest.raises(ValueError):
        humanize.naturaldelta(dt.timedelta(seconds=5), minimum_unit="hours")
    with pytest.raises(KeyError):
        humanize.naturaldelta(dt.timedelta(seconds=5), minimum_unit="fortnights")
    with pytest.raises(OverflowError):
        humanize.naturaldelta(1e30)


# ---------------------------------------------------------------- naturaltime


def test_naturaltime_with_when_past_and_future():
    assert humanize.naturaltime(WHEN - dt.timedelta(seconds=30), when=WHEN) == (
        "30 seconds ago"
    )
    assert humanize.naturaltime(WHEN - dt.timedelta(minutes=5), when=WHEN) == (
        "5 minutes ago"
    )
    assert humanize.naturaltime(WHEN - dt.timedelta(days=5), when=WHEN) == "5 days ago"
    assert humanize.naturaltime(WHEN + dt.timedelta(hours=3), when=WHEN) == (
        "3 hours from now"
    )
    assert humanize.naturaltime(WHEN + dt.timedelta(days=2), when=WHEN) == (
        "2 days from now"
    )


def test_naturaltime_now_and_plain_numbers():
    assert humanize.naturaltime(WHEN, when=WHEN) == "now"
    assert humanize.naturaltime(0) == "now"
    assert humanize.naturaltime(30) == "30 seconds ago"
    assert humanize.naturaltime(30, future=True) == "30 seconds from now"
    assert humanize.naturaltime(3600, when=WHEN) == "an hour ago"
    assert humanize.naturaltime(3600, future=True, when=WHEN) == "an hour from now"


def test_naturaltime_timedelta_tense_comes_from_sign():
    assert humanize.naturaltime(dt.timedelta(minutes=5), when=WHEN) == "5 minutes ago"
    assert humanize.naturaltime(dt.timedelta(minutes=-5), when=WHEN) == (
        "5 minutes from now"
    )


def test_naturaltime_months_and_minimum_unit_options():
    value = WHEN - dt.timedelta(days=60)
    assert humanize.naturaltime(value, when=WHEN) == "2 months ago"
    assert humanize.naturaltime(value, months=False, when=WHEN) == "60 days ago"
    short = WHEN - dt.timedelta(milliseconds=500)
    assert humanize.naturaltime(short, when=WHEN) == "now"
    assert humanize.naturaltime(short, minimum_unit="milliseconds", when=WHEN) == (
        "500 milliseconds ago"
    )


def test_naturaltime_invalid_values():
    assert humanize.naturaltime("foo") == "foo"
    assert humanize.naturaltime(None) == "None"


def test_naturaltime_aware_datetimes_with_when():
    utc = dt.timezone.utc
    when = dt.datetime(2024, 6, 15, 12, 0, 0, tzinfo=utc)
    assert humanize.naturaltime(when - dt.timedelta(hours=2), when=when) == "2 hours ago"
    plus5 = dt.timezone(dt.timedelta(hours=5))
    same_instant = dt.datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
    assert humanize.naturaltime(same_instant, when=when) == "now"


def test_naturaltime_relative_to_now_past():
    assert humanize.naturaltime(dt.datetime.now() - dt.timedelta(seconds=30)) == (
        "30 seconds ago"
    )
    assert humanize.naturaltime(dt.datetime.now() - dt.timedelta(minutes=10)) == (
        "10 minutes ago"
    )
    assert humanize.naturaltime(dt.datetime.now() - dt.timedelta(hours=2)) == "2 hours ago"
    assert humanize.naturaltime(dt.datetime.now() - dt.timedelta(days=3)) == "3 days ago"
    assert humanize.naturaltime(dt.datetime.now()) == "now"


def test_naturaltime_relative_to_now_future():
    assert humanize.naturaltime(dt.datetime.now() + dt.timedelta(hours=5, minutes=1)) == (
        "5 hours from now"
    )
    assert humanize.naturaltime(dt.datetime.now() + dt.timedelta(days=2, hours=1)) == (
        "2 days from now"
    )


@freeze_time("2024-03-15 10:30:00")
def test_naturaltime_with_frozen_time():
    now = dt.datetime.now()
    assert humanize.naturaltime(now - dt.timedelta(minutes=1)) == "a minute ago"
    assert humanize.naturaltime(now - dt.timedelta(hours=1)) == "an hour ago"
    assert humanize.naturaltime(now + dt.timedelta(days=1)) == "a day from now"
    assert humanize.naturaltime(dt.datetime(2024, 3, 15, 10, 29, 30)) == "30 seconds ago"


# ---------------------------------------------------------------- naturalday


def test_naturalday_today_tomorrow_yesterday():
    today = dt.date.today()
    assert humanize.naturalday(today) == "today"
    assert humanize.naturalday(today + dt.timedelta(days=1)) == "tomorrow"
    assert humanize.naturalday(today - dt.timedelta(days=1)) == "yesterday"


def test_naturalday_accepts_datetimes():
    assert humanize.naturalday(dt.datetime.now()) == "today"
    assert humanize.naturalday(dt.datetime.now() + dt.timedelta(days=1)) == "tomorrow"
    assert humanize.naturalday(dt.datetime.now(dt.timezone.utc)) == "today"


def test_naturalday_far_dates_and_non_dates():
    value = dt.date.today() + dt.timedelta(days=30)
    assert humanize.naturalday(value) == value.strftime("%b %d")
    assert humanize.naturalday(value, "%Y-%m-%d") == value.isoformat()
    assert humanize.naturalday("foo") == "foo"
    assert humanize.naturalday(None) == "None"
    assert humanize.naturalday(12345) == "12345"


@freeze_time("2024-03-15 08:00:00")
def test_naturalday_frozen():
    assert humanize.naturalday(dt.date(2024, 3, 15)) == "today"
    assert humanize.naturalday(dt.date(2024, 3, 16)) == "tomorrow"
    assert humanize.naturalday(dt.date(2024, 3, 14)) == "yesterday"
    assert humanize.naturalday(dt.date(2024, 3, 20)) == "Mar 20"
    assert humanize.naturalday(dt.date(2023, 1, 5), "%d/%m/%Y") == "05/01/2023"


# ---------------------------------------------------------------- naturaldate


@freeze_time("2024-03-15 08:00:00")
def test_naturaldate_frozen():
    assert humanize.naturaldate(dt.date(2024, 3, 15)) == "today"
    assert humanize.naturaldate(dt.date(2024, 3, 14)) == "yesterday"
    assert humanize.naturaldate(dt.date(2024, 3, 16)) == "tomorrow"
    assert humanize.naturaldate(dt.date(2024, 3, 20)) == "Mar 20"
    assert humanize.naturaldate(dt.date(2024, 8, 13)) == "Aug 13"
    assert humanize.naturaldate(dt.date(2024, 8, 20)) == "Aug 20 2024"
    assert humanize.naturaldate(dt.date(2023, 3, 15)) == "Mar 15 2023"
    assert humanize.naturaldate(dt.datetime(2025, 1, 2, 3, 4)) == "Jan 02 2025"


def test_naturaldate_relative_to_now_and_non_dates():
    today = dt.date.today()
    assert humanize.naturaldate(today) == "today"
    assert humanize.naturaldate(today + dt.timedelta(days=1)) == "tomorrow"
    far = today + dt.timedelta(days=400)
    assert humanize.naturaldate(far) == far.strftime("%b %d %Y")
    assert humanize.naturaldate("foo") == "foo"
    assert humanize.naturaldate(None) == "None"


# ---------------------------------------------------------------- precisedelta


@pytest.fixture
def sample_delta():
    return dt.timedelta(seconds=3633, days=2, microseconds=123000)


def test_precisedelta_default_and_custom_format(sample_delta):
    assert humanize.precisedelta(sample_delta) == "2 days, 1 hour and 33.12 seconds"
    assert humanize.precisedelta(sample_delta, format="%0.4f") == (
        "2 days, 1 hour and 33.1230 seconds"
    )


def test_precisedelta_finer_minimum_units(sample_delta):
    assert humanize.precisedelta(sample_delta, minimum_unit="microseconds") == (
        "2 days, 1 hour, 33 seconds and 123 milliseconds"
    )
    assert humanize.precisedelta(
        dt.timedelta(milliseconds=1500), minimum_unit="milliseconds"
    ) == "1 second and 500 milliseconds"


def test_precisedelta_suppress(sample_delta):
    assert humanize.precisedelta(sample_delta, suppress=["days"]) == (
        "49 hours and 33.12 seconds"
    )
    assert humanize.precisedelta(
        dt.timedelta(seconds=90, microseconds=100),
        suppress=["seconds", "milliseconds", "microseconds"],
    ) == "1.50 minutes"
    assert humanize.precisedelta(dt.timedelta(days=400), suppress=["years"]) == (
        "13 months and 3 days"
    )


def test_precisedelta_minimum_unit_minutes():
    assert humanize.precisedelta(dt.timedelta(seconds=1), minimum_unit="minutes") == (
        "0.02 minutes"
    )
    assert humanize.precisedelta(dt.timedelta(seconds=0.1), minimum_unit="minutes") == (
        "0 minutes"
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        (dt.timedelta(seconds=1), "1 second"),
        (dt.timedelta(seconds=90), "1 minute and 30 seconds"),
        (dt.timedelta(hours=1, minutes=1, seconds=1), "1 hour, 1 minute and 1 second"),
        (3661, "1 hour, 1 minute and 1 second"),
        (dt.timedelta(days=2), "2 days"),
        (dt.timedelta(days=365), "1 year"),
        (dt.timedelta(days=35), "1 month and 4 days"),
        (dt.timedelta(days=400), "1 year, 1 month and 4 days"),
        (dt.timedelta(seconds=-90), "1 minute and 30 seconds"),
        (0, "0 seconds"),
    ],
)
def test_precisedelta_various(value, expected):
    assert humanize.precisedelta(value) == expected


def test_precisedelta_invalid_inputs():
    assert humanize.precisedelta("foo") == "foo"
    assert humanize.precisedelta(None) == "None"
    with pytest.raises(KeyError):
        humanize.precisedelta(dt.timedelta(seconds=5), minimum_unit="bogus")
    with pytest.raises(ValueError):
        humanize.precisedelta(
            dt.timedelta(days=3), minimum_unit="years", suppress=["years"]
        )
