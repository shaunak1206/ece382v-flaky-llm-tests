"""Tests for humanize.time module."""

import datetime as dt
import pytest
from freezegun import freeze_time
from humanize import naturaldelta, naturaltime, naturalday, naturaldate, precisedelta


class TestNaturaldelta:
    """Tests for naturaldelta function."""

    def test_naturaldelta_seconds(self):
        delta = dt.timedelta(seconds=30)
        assert "30 second" in naturaldelta(delta)

    def test_naturaldelta_one_second(self):
        delta = dt.timedelta(seconds=1)
        result = naturaldelta(delta)
        assert "second" in result

    def test_naturaldelta_moment(self):
        delta = dt.timedelta(seconds=0.5)
        result = naturaldelta(delta)
        assert result is not None

    def test_naturaldelta_minutes(self):
        delta = dt.timedelta(minutes=5)
        assert "5 minute" in naturaldelta(delta)

    def test_naturaldelta_one_minute(self):
        delta = dt.timedelta(minutes=1)
        result = naturaldelta(delta)
        assert "minute" in result

    def test_naturaldelta_hours(self):
        delta = dt.timedelta(hours=2)
        assert "2 hour" in naturaldelta(delta)

    def test_naturaldelta_one_hour(self):
        delta = dt.timedelta(hours=1)
        result = naturaldelta(delta)
        assert "hour" in result

    def test_naturaldelta_days(self):
        delta = dt.timedelta(days=7)
        result = naturaldelta(delta)
        assert "day" in result or "month" in result

    def test_naturaldelta_one_day(self):
        delta = dt.timedelta(days=1)
        result = naturaldelta(delta)
        assert "day" in result

    def test_naturaldelta_float_seconds(self):
        result = naturaldelta(120.5)
        assert result is not None

    def test_naturaldelta_with_months(self):
        delta = dt.timedelta(days=45)
        result = naturaldelta(delta, months=True)
        assert result is not None

    def test_naturaldelta_minimum_unit_milliseconds(self):
        delta = dt.timedelta(milliseconds=500)
        result = naturaldelta(delta, minimum_unit="milliseconds")
        assert "millisecond" in result or "500" in result

    def test_naturaldelta_invalid(self):
        result = naturaldelta("invalid")
        assert result == "invalid"


class TestNaturaltime:
    """Tests for naturaltime function."""

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_now(self):
        now = dt.datetime(2023, 1, 15, 12, 0, 0)
        result = naturaltime(now)
        assert "now" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_past(self):
        past = dt.datetime(2023, 1, 15, 11, 0, 0)
        result = naturaltime(past)
        assert "ago" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_future(self):
        future = dt.datetime(2023, 1, 15, 13, 0, 0)
        result = naturaltime(future)
        assert "from now" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_timedelta(self):
        delta = dt.timedelta(hours=2)
        result = naturaltime(delta)
        assert result is not None

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_with_seconds_float(self):
        result = naturaltime(3600.0)
        assert "hour" in result or "ago" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_when_parameter(self):
        past = dt.datetime(2023, 1, 15, 10, 0, 0)
        when = dt.datetime(2023, 1, 15, 12, 0, 0)
        result = naturaltime(past, when=when)
        assert "ago" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_naturaltime_minimum_unit(self):
        delta = dt.timedelta(milliseconds=500)
        result = naturaltime(delta, minimum_unit="milliseconds")
        assert result is not None


class TestNaturalday:
    """Tests for naturalday function."""

    @freeze_time("2023-01-15")
    def test_naturalday_today(self):
        today = dt.date(2023, 1, 15)
        assert naturalday(today) == "today"

    @freeze_time("2023-01-15")
    def test_naturalday_tomorrow(self):
        tomorrow = dt.date(2023, 1, 16)
        assert naturalday(tomorrow) == "tomorrow"

    @freeze_time("2023-01-15")
    def test_naturalday_yesterday(self):
        yesterday = dt.date(2023, 1, 14)
        assert naturalday(yesterday) == "yesterday"

    @freeze_time("2023-01-15")
    def test_naturalday_other(self):
        other_day = dt.date(2023, 1, 20)
        result = naturalday(other_day)
        assert "Jan" in result or "20" in result

    @freeze_time("2023-01-15")
    def test_naturalday_with_datetime(self):
        today_dt = dt.datetime(2023, 1, 15, 12, 0, 0)
        assert naturalday(today_dt) == "today"

    @freeze_time("2023-01-15")
    def test_naturalday_custom_format(self):
        other_day = dt.date(2023, 1, 20)
        result = naturalday(other_day, "%Y-%m-%d")
        assert "2023-01-20" in result

    @freeze_time("2023-01-15")
    def test_naturalday_invalid(self):
        result = naturalday("invalid")
        assert result == "invalid"


class TestNaturaldate:
    """Tests for naturaldate function."""

    @freeze_time("2023-01-15")
    def test_naturaldate_today(self):
        today = dt.date(2023, 1, 15)
        result = naturaldate(today)
        assert "today" in result

    @freeze_time("2023-01-15")
    def test_naturaldate_tomorrow(self):
        tomorrow = dt.date(2023, 1, 16)
        result = naturaldate(tomorrow)
        assert "tomorrow" in result

    @freeze_time("2023-01-15")
    def test_naturaldate_far_future(self):
        far_future = dt.date(2023, 8, 15)
        result = naturaldate(far_future)
        assert "2023" in result

    @freeze_time("2023-01-15")
    def test_naturaldate_recent_future(self):
        future = dt.date(2023, 1, 20)
        result = naturaldate(future)
        assert result is not None

    @freeze_time("2023-01-15")
    def test_naturaldate_datetime(self):
        dt_obj = dt.datetime(2023, 1, 15, 10, 30, 0)
        result = naturaldate(dt_obj)
        assert "today" in result

    @freeze_time("2023-01-15")
    def test_naturaldate_invalid(self):
        result = naturaldate("invalid")
        assert result == "invalid"


class TestPrecisedelta:
    """Tests for precisedelta function."""

    def test_precisedelta_basic(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        result = precisedelta(delta)
        assert "day" in result and "hour" in result

    def test_precisedelta_custom_format(self):
        delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
        result = precisedelta(delta, format="%0.4f")
        assert result is not None

    def test_precisedelta_minimum_unit_microseconds(self):
        delta = dt.timedelta(seconds=33, microseconds=123000)
        result = precisedelta(delta, minimum_unit="microseconds")
        assert result is not None

    def test_precisedelta_suppress_units(self):
        delta = dt.timedelta(seconds=3633, days=2)
        result = precisedelta(delta, suppress=["days"])
        assert "hour" in result
        assert "day" not in result

    def test_precisedelta_small_value(self):
        delta = dt.timedelta(seconds=0.1)
        result = precisedelta(delta, minimum_unit="minutes")
        assert "0" in result

    def test_precisedelta_float_seconds(self):
        result = precisedelta(3600.5)
        assert result is not None

    def test_precisedelta_invalid(self):
        result = precisedelta("invalid")
        assert result == "invalid"

    def test_precisedelta_one_minute(self):
        delta = dt.timedelta(minutes=1, seconds=0)
        result = precisedelta(delta)
        assert "minute" in result

    def test_precisedelta_complex(self):
        # 1 year, 2 months, 3 days, 4 hours, 5 minutes, 6 seconds
        delta = dt.timedelta(
            days=365+61,
            hours=4,
            minutes=5,
            seconds=6
        )
        result = precisedelta(delta)
        assert result is not None


class TestNaturalDeltaEdgeCases:
    """Tests for edge cases and additional scenarios."""

    def test_naturaldelta_zero_seconds(self):
        delta = dt.timedelta(seconds=0)
        result = naturaldelta(delta)
        assert result is not None

    def test_naturaldelta_negative_delta(self):
        delta = dt.timedelta(seconds=-60)
        result = naturaldelta(delta)
        assert "minute" in result

    def test_naturaldelta_milliseconds(self):
        delta = dt.timedelta(milliseconds=500)
        result = naturaldelta(delta, minimum_unit="milliseconds")
        assert "millisecond" in result or result is not None

    def test_naturaldelta_invalid_minimum_unit(self):
        delta = dt.timedelta(seconds=60)
        with pytest.raises(ValueError):
            naturaldelta(delta, minimum_unit="invalid")

    def test_naturaldelta_months_false(self):
        delta = dt.timedelta(days=45)
        result = naturaldelta(delta, months=False)
        assert "day" in result
