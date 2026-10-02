"""Tests for arrow API and factory methods."""

import datetime as dt
from datetime import timezone, timedelta, date
import pytest
import arrow
from freezegun import freeze_time


class TestArrowGet:
    """Tests for arrow.get() function."""

    def test_get_no_args(self):
        """Test get() with no arguments returns current time."""
        result = arrow.get()
        assert isinstance(result, arrow.Arrow)

    def test_get_iso_string(self):
        """Test get() with ISO format string."""
        result = arrow.get("2020-01-01")
        assert result.year == 2020
        assert result.month == 1
        assert result.day == 1

    def test_get_iso_datetime_string(self):
        """Test get() with full ISO datetime string."""
        result = arrow.get("2020-05-15T10:30:45Z")
        assert result.year == 2020
        assert result.month == 5
        assert result.day == 15

    def test_get_datetime_object(self):
        """Test get() with datetime object."""
        dt_obj = dt.datetime(2020, 5, 15, 10, 30, 45)
        result = arrow.get(dt_obj)
        assert result.year == 2020

    def test_get_date_object(self):
        """Test get() with date object."""
        d = dt.date(2020, 5, 15)
        result = arrow.get(d)
        assert result.year == 2020
        assert result.month == 5
        assert result.day == 15

    def test_get_timestamp(self):
        """Test get() with timestamp."""
        result = arrow.get(1589462745)
        assert isinstance(result, arrow.Arrow)

    def test_get_float_timestamp(self):
        """Test get() with float timestamp."""
        result = arrow.get(1589462745.5)
        assert isinstance(result, arrow.Arrow)

    def test_get_arrow_object(self):
        """Test get() with Arrow object."""
        arw = arrow.get("2020-05-15")
        result = arrow.get(arw)
        assert result.year == 2020

    def test_get_year_month_day(self):
        """Test get() with year, month, day integers."""
        result = arrow.get(2020, 5, 15)
        assert result.year == 2020
        assert result.month == 5
        assert result.day == 15

    def test_get_with_timezone(self):
        """Test get() with timezone."""
        result = arrow.get("2020-05-15", tzinfo="US/Pacific")
        assert result.year == 2020

    def test_get_two_datetimes_with_tz(self):
        """Test get() with datetime and timezone."""
        dt_obj = dt.datetime(2020, 5, 15, 10, 30)
        result = arrow.get(dt_obj, "US/Eastern")
        assert result.year == 2020

    def test_get_string_with_format(self):
        """Test get() with string and format."""
        result = arrow.get("2020-05-15", "YYYY-MM-DD")
        assert result.year == 2020

    def test_get_multiple_formats(self):
        """Test get() with multiple possible formats."""
        result = arrow.get("2020-05-15", ["YYYY-MM-DD", "YYYY/MM/DD"])
        assert result.year == 2020


class TestArrowNow:
    """Tests for arrow.now() function."""

    def test_now_returns_arrow(self):
        """Test that now() returns an Arrow object."""
        result = arrow.now()
        assert isinstance(result, arrow.Arrow)

    def test_now_is_current_time(self):
        """Test that now() returns current time."""
        before = arrow.utcnow()
        result = arrow.now()
        after = arrow.utcnow()
        assert before <= result <= after or result >= before

    def test_now_with_timezone(self):
        """Test now() with timezone."""
        result = arrow.now("US/Pacific")
        assert isinstance(result, arrow.Arrow)

    def test_now_utc(self):
        """Test now() with UTC."""
        result = arrow.now("UTC")
        assert isinstance(result, arrow.Arrow)

    @freeze_time("2023-01-15 12:00:00")
    def test_now_frozen_time(self):
        """Test now() with frozen time."""
        result = arrow.now()
        assert result.year == 2023
        assert result.month == 1


class TestArrowUtcnow:
    """Tests for arrow.utcnow() function."""

    def test_utcnow_returns_arrow(self):
        """Test that utcnow() returns an Arrow object."""
        result = arrow.utcnow()
        assert isinstance(result, arrow.Arrow)

    def test_utcnow_is_utc(self):
        """Test that utcnow() returns UTC time."""
        result = arrow.utcnow()
        assert result.tzinfo.zone == "UTC" or str(result.tzinfo) == "UTC+00:00"

    @freeze_time("2023-01-15 12:30:45")
    def test_utcnow_frozen_time(self):
        """Test utcnow() with frozen time."""
        result = arrow.utcnow()
        assert result.hour == 12


class TestArrowConstructor:
    """Tests for Arrow constructor."""

    def test_arrow_constructor_basic(self):
        """Test basic Arrow constructor."""
        arw = arrow.Arrow(2020, 5, 15)
        assert arw.year == 2020
        assert arw.month == 5
        assert arw.day == 15

    def test_arrow_constructor_with_time(self):
        """Test Arrow constructor with time."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        assert arw.hour == 10
        assert arw.minute == 30
        assert arw.second == 45

    def test_arrow_constructor_with_microseconds(self):
        """Test Arrow constructor with microseconds."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45, 123456)
        assert arw.microsecond == 123456

    def test_arrow_constructor_with_tz(self):
        """Test Arrow constructor with timezone."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="US/Eastern")
        assert arw.year == 2020

    def test_arrow_constructor_default_tz_is_utc(self):
        """Test that default timezone is UTC."""
        arw = arrow.Arrow(2020, 5, 15)
        assert "UTC" in str(arw.tzinfo)


class TestArrowClassMethods:
    """Tests for Arrow class methods."""

    def test_fromdatetime(self):
        """Test fromdatetime class method."""
        dt_obj = dt.datetime(2020, 5, 15, 10, 30, 45)
        arw = arrow.Arrow.fromdatetime(dt_obj)
        assert arw.year == 2020
        assert arw.month == 5

    def test_fromdatetime_with_tz(self):
        """Test fromdatetime with timezone."""
        dt_obj = dt.datetime(2020, 5, 15, 10, 30)
        arw = arrow.Arrow.fromdatetime(dt_obj, "US/Eastern")
        assert arw.year == 2020

    def test_fromdate(self):
        """Test fromdate class method."""
        d = dt.date(2020, 5, 15)
        arw = arrow.Arrow.fromdate(d)
        assert arw.year == 2020
        assert arw.day == 15
        assert arw.hour == 0

    def test_fromtimestamp(self):
        """Test fromtimestamp class method."""
        arw = arrow.Arrow.fromtimestamp(1589462745)
        assert isinstance(arw, arrow.Arrow)

    def test_utcfromtimestamp(self):
        """Test utcfromtimestamp class method."""
        arw = arrow.Arrow.utcfromtimestamp(1589462745)
        assert isinstance(arw, arrow.Arrow)

    def test_strptime(self):
        """Test strptime class method."""
        arw = arrow.Arrow.strptime("2020-05-15", "%Y-%m-%d")
        assert arw.year == 2020
        assert arw.month == 5

    def test_fromordinal(self):
        """Test fromordinal class method."""
        arw = arrow.Arrow.fromordinal(737741)
        assert isinstance(arw, arrow.Arrow)


class TestArrowProperties:
    """Tests for Arrow properties."""

    def test_year_property(self):
        """Test year property."""
        arw = arrow.Arrow(2020, 5, 15)
        assert arw.year == 2020

    def test_month_property(self):
        """Test month property."""
        arw = arrow.Arrow(2020, 5, 15)
        assert arw.month == 5

    def test_day_property(self):
        """Test day property."""
        arw = arrow.Arrow(2020, 5, 15)
        assert arw.day == 15

    def test_hour_property(self):
        """Test hour property."""
        arw = arrow.Arrow(2020, 5, 15, 10)
        assert arw.hour == 10

    def test_minute_property(self):
        """Test minute property."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        assert arw.minute == 30

    def test_second_property(self):
        """Test second property."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        assert arw.second == 45

    def test_tzinfo_property(self):
        """Test tzinfo property."""
        arw = arrow.Arrow(2020, 5, 15)
        assert arw.tzinfo is not None

    def test_datetime_property(self):
        """Test datetime property."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        dt_obj = arw.datetime
        assert isinstance(dt_obj, dt.datetime)
        assert dt_obj.year == 2020

    def test_timestamp(self):
        """Test timestamp method."""
        arw = arrow.Arrow(2020, 1, 1, 0, 0, 0)
        ts = arw.timestamp()
        assert isinstance(ts, float)

    def test_int_timestamp(self):
        """Test int_timestamp property."""
        arw = arrow.Arrow(2020, 1, 1)
        ts = arw.int_timestamp
        assert isinstance(ts, int)
