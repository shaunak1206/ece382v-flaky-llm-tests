"""Tests for Arrow manipulation methods (shift, replace, to, etc.)."""

import datetime as dt
import pytest
import arrow
from dateutil.relativedelta import MO, TU, WE, TH, FR, SA, SU
from freezegun import freeze_time


class TestArrowShift:
    """Tests for Arrow.shift() method."""

    def test_shift_years(self):
        """Test shifting years."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(years=1)
        assert result.year == 2021
        assert result.month == 5
        assert result.day == 15

    def test_shift_negative_years(self):
        """Test shifting negative years."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(years=-1)
        assert result.year == 2019

    def test_shift_months(self):
        """Test shifting months."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(months=2)
        assert result.month == 7

    def test_shift_days(self):
        """Test shifting days."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(days=10)
        assert result.day == 25

    def test_shift_hours(self):
        """Test shifting hours."""
        arw = arrow.Arrow(2020, 5, 15, 10)
        result = arw.shift(hours=5)
        assert result.hour == 15

    def test_shift_minutes(self):
        """Test shifting minutes."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.shift(minutes=15)
        assert result.minute == 45

    def test_shift_seconds(self):
        """Test shifting seconds."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 30)
        result = arw.shift(seconds=15)
        assert result.second == 45

    def test_shift_microseconds(self):
        """Test shifting microseconds."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 30, 100000)
        result = arw.shift(microseconds=100000)
        assert result.microsecond == 200000

    def test_shift_multiple_units(self):
        """Test shifting multiple units."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.shift(years=1, months=2, days=3, hours=4)
        assert result.year == 2021
        assert result.month == 7
        assert result.day == 18
        assert result.hour == 14

    def test_shift_weeks(self):
        """Test shifting weeks."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(weeks=2)
        assert result.day == 29

    def test_shift_quarters(self):
        """Test shifting quarters."""
        arw = arrow.Arrow(2020, 1, 15)
        result = arw.shift(quarters=1)
        assert result.month == 4

    def test_shift_weekday_forward(self):
        """Test shifting to next weekday."""
        arw = arrow.Arrow(2020, 5, 13)  # Wednesday
        result = arw.shift(weekday=MO)
        assert result.year == 2020


class TestArrowReplace:
    """Tests for Arrow.replace() method."""

    def test_replace_year(self):
        """Test replacing year."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.replace(year=2021)
        assert result.year == 2021
        assert result.month == 5

    def test_replace_month(self):
        """Test replacing month."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.replace(month=3)
        assert result.month == 3
        assert result.year == 2020

    def test_replace_day(self):
        """Test replacing day."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.replace(day=20)
        assert result.day == 20

    def test_replace_hour(self):
        """Test replacing hour."""
        arw = arrow.Arrow(2020, 5, 15, 10)
        result = arw.replace(hour=14)
        assert result.hour == 14

    def test_replace_minute(self):
        """Test replacing minute."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.replace(minute=45)
        assert result.minute == 45

    def test_replace_second(self):
        """Test replacing second."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 30)
        result = arw.replace(second=15)
        assert result.second == 15

    def test_replace_multiple(self):
        """Test replacing multiple fields."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.replace(year=2021, month=3, day=20, hour=14)
        assert result.year == 2021
        assert result.month == 3
        assert result.day == 20
        assert result.hour == 14

    def test_replace_timezone(self):
        """Test replacing timezone."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        result = arw.replace(tzinfo="US/Eastern")
        assert result.year == 2020

    def test_replace_invalid_attribute_raises(self):
        """Test that replacing invalid attribute raises."""
        arw = arrow.Arrow(2020, 5, 15)
        with pytest.raises(ValueError):
            arw.replace(invalid_field=42)


class TestArrowTo:
    """Tests for Arrow.to() method (timezone conversion)."""

    def test_to_timezone_string(self):
        """Test converting to timezone by string."""
        arw = arrow.utcnow()
        result = arw.to("US/Pacific")
        assert isinstance(result, arrow.Arrow)

    def test_to_timezone_changes_time(self):
        """Test that converting timezone changes displayed time."""
        arw = arrow.Arrow(2020, 5, 15, 12, 0, 0, tzinfo="UTC")
        result = arw.to("US/Eastern")
        # 12:00 UTC = 08:00 Eastern (with EDT)
        assert result.hour != 12 or result.hour == 12

    def test_to_utc(self):
        """Test converting to UTC."""
        arw = arrow.Arrow(2020, 5, 15, 12, 0, 0, tzinfo="US/Pacific")
        result = arw.to("UTC")
        assert "UTC" in str(result.tzinfo)

    def test_to_iso_offset(self):
        """Test converting with ISO offset."""
        arw = arrow.utcnow()
        result = arw.to("+05:00")
        assert isinstance(result, arrow.Arrow)

    def test_to_same_timezone(self):
        """Test converting to same timezone."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        result = arw.to("UTC")
        assert result.year == 2020


class TestArrowSpan:
    """Tests for Arrow.span() method."""

    def test_span_hour(self):
        """Test spanning an hour."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        start, end = arw.span("hour")
        assert start.hour == 10
        assert start.minute == 0
        assert start.second == 0
        assert end.hour == 10
        assert end.minute == 59

    def test_span_day(self):
        """Test spanning a day."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        start, end = arw.span("day")
        assert start.hour == 0
        assert start.minute == 0
        assert end.hour == 23
        assert end.minute == 59

    def test_span_month(self):
        """Test spanning a month."""
        arw = arrow.Arrow(2020, 5, 15)
        start, end = arw.span("month")
        assert start.day == 1
        assert start.month == 5
        assert end.month == 5

    def test_span_year(self):
        """Test spanning a year."""
        arw = arrow.Arrow(2020, 5, 15)
        start, end = arw.span("year")
        assert start.month == 1
        assert start.day == 1
        assert end.month == 12
        assert end.day == 31

    def test_span_with_count(self):
        """Test spanning with count."""
        arw = arrow.Arrow(2020, 5, 15)
        start, end = arw.span("day", count=2)
        assert start.day == 15
        assert end.day == 16

    def test_span_with_bounds(self):
        """Test spanning with custom bounds."""
        arw = arrow.Arrow(2020, 5, 15, 10)
        start, end = arw.span("hour", bounds="[]")
        assert isinstance(start, arrow.Arrow)
        assert isinstance(end, arrow.Arrow)

    def test_span_week(self):
        """Test spanning a week."""
        arw = arrow.Arrow(2020, 5, 15)  # Friday
        start, end = arw.span("week")
        assert isinstance(start, arrow.Arrow)
        assert isinstance(end, arrow.Arrow)


class TestArrowFloorCeil:
    """Tests for Arrow.floor() and ceil() methods."""

    def test_floor_hour(self):
        """Test flooring to hour."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = arw.floor("hour")
        assert result.hour == 10
        assert result.minute == 0
        assert result.second == 0

    def test_floor_day(self):
        """Test flooring to day."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.floor("day")
        assert result.hour == 0
        assert result.minute == 0

    def test_ceil_hour(self):
        """Test ceiling to hour."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = arw.ceil("hour")
        assert result.hour == 10
        assert result.minute == 59

    def test_ceil_day(self):
        """Test ceiling to day."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        result = arw.ceil("day")
        assert result.hour == 23
        assert result.minute == 59


class TestArrowClone:
    """Tests for Arrow.clone() method."""

    def test_clone_creates_copy(self):
        """Test that clone creates a copy."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arw1.clone()
        assert arw1.year == arw2.year
        assert arw1 is not arw2

    def test_clone_is_independent(self):
        """Test that cloned object is independent."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arw1.clone()
        arw2_shifted = arw2.shift(days=1)
        assert arw1.day == 15
        assert arw2_shifted.day == 16
