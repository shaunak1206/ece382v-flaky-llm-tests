"""Tests for Arrow range, span_range, interval, and formatting methods."""

import datetime as dt
import pytest
import arrow
from freezegun import freeze_time


class TestArrowRange:
    """Tests for Arrow.range() class method."""

    def test_range_hour(self):
        """Test range by hour."""
        start = dt.datetime(2020, 5, 5, 12, 0)
        end = dt.datetime(2020, 5, 5, 15, 0)
        result = list(arrow.Arrow.range("hour", start, end))
        assert len(result) >= 3
        assert result[0].hour == 12

    def test_range_day(self):
        """Test range by day."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 10)
        result = list(arrow.Arrow.range("day", start, end))
        assert len(result) >= 5
        assert result[0].day == 5

    def test_range_with_limit(self):
        """Test range with limit."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 31)
        result = list(arrow.Arrow.range("day", start, limit=5))
        assert len(result) <= 5

    def test_range_minute(self):
        """Test range by minute."""
        start = dt.datetime(2020, 5, 5, 12, 0)
        end = dt.datetime(2020, 5, 5, 12, 10)
        result = list(arrow.Arrow.range("minute", start, end))
        assert len(result) >= 10

    def test_range_with_tz(self):
        """Test range with timezone."""
        start = arrow.Arrow(2020, 5, 5, tzinfo="UTC")
        end = arrow.Arrow(2020, 5, 6, tzinfo="UTC")
        result = list(arrow.Arrow.range("hour", start, end, limit=5))
        assert len(result) > 0

    def test_range_end_included(self):
        """Test that range end point can be included."""
        start = dt.datetime(2020, 5, 5, 12, 0)
        end = dt.datetime(2020, 5, 5, 13, 0)
        result = list(arrow.Arrow.range("hour", start, end))
        # End point may be included
        assert len(result) >= 1


class TestArrowSpanRange:
    """Tests for Arrow.span_range() class method."""

    def test_span_range_hour(self):
        """Test span_range by hour."""
        start = dt.datetime(2020, 5, 5, 12, 0)
        end = dt.datetime(2020, 5, 5, 15, 0)
        result = list(arrow.Arrow.span_range("hour", start, end))
        assert len(result) > 0
        start_span, end_span = result[0]
        assert isinstance(start_span, arrow.Arrow)
        assert isinstance(end_span, arrow.Arrow)

    def test_span_range_day(self):
        """Test span_range by day."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 10)
        result = list(arrow.Arrow.span_range("day", start, end))
        assert len(result) > 0

    def test_span_range_with_limit(self):
        """Test span_range with limit."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 31)
        result = list(arrow.Arrow.span_range("day", start, end, limit=5))
        assert len(result) <= 5

    def test_span_range_with_bounds(self):
        """Test span_range with custom bounds."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 10)
        result = list(arrow.Arrow.span_range("day", start, end, bounds="[]"))
        assert len(result) > 0


class TestArrowInterval:
    """Tests for Arrow.interval() class method."""

    def test_interval_hour(self):
        """Test interval by hour."""
        start = dt.datetime(2020, 5, 5, 12, 0)
        end = dt.datetime(2020, 5, 5, 20, 0)
        result = list(arrow.Arrow.interval("hour", start, end, interval=2))
        assert len(result) > 0

    def test_interval_day(self):
        """Test interval by day."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 20)
        result = list(arrow.Arrow.interval("day", start, end, interval=5))
        assert len(result) > 0

    def test_interval_with_limit(self):
        """Test interval with limit."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 6, 30)
        result = list(arrow.Arrow.interval("day", start, end, interval=10, limit=3))
        assert len(result) <= 3

    def test_interval_invalid_interval_raises(self):
        """Test that invalid interval raises."""
        start = dt.datetime(2020, 5, 5)
        end = dt.datetime(2020, 5, 10)
        with pytest.raises(ValueError):
            list(arrow.Arrow.interval("day", start, end, interval=0))


class TestArrowFormat:
    """Tests for Arrow.format() method."""

    def test_format_default(self):
        """Test format with default format string."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 47)
        result = arw.format()
        assert "2020" in result

    def test_format_iso(self):
        """Test format with ISO format."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.format("YYYY-MM-DD")
        assert result == "2020-05-15"

    def test_format_with_time(self):
        """Test format with time."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = arw.format("YYYY-MM-DD HH:mm:ss")
        assert "2020-05-15" in result
        assert "10:30:45" in result

    def test_format_month_name(self):
        """Test format with month name."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.format("MMMM DD, YYYY")
        assert "May" in result or "05" in result

    def test_format_timestamp(self):
        """Test format as timestamp."""
        arw = arrow.Arrow(2020, 1, 1, 0, 0, 0, tzinfo="UTC")
        result = arw.format("X")
        assert result is not None

    def test_format_with_locale(self):
        """Test format with locale."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.format("MMMM", locale="en_us")
        assert result is not None


class TestArrowIsoformat:
    """Tests for Arrow.isoformat() method."""

    def test_isoformat_returns_string(self):
        """Test that isoformat returns string."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = arw.isoformat()
        assert isinstance(result, str)

    def test_isoformat_includes_date(self):
        """Test that isoformat includes date."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.isoformat()
        assert "2020-05-15" in result

    def test_isoformat_includes_time(self):
        """Test that isoformat includes time."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = arw.isoformat()
        assert "10:30:45" in result


class TestArrowDatetimeProperties:
    """Tests for Arrow datetime-like properties and methods."""

    def test_weekday(self):
        """Test weekday property."""
        arw = arrow.Arrow(2020, 5, 13)  # Wednesday
        assert arw.weekday() == 2

    def test_isoweekday(self):
        """Test isoweekday method."""
        arw = arrow.Arrow(2020, 5, 13)  # Wednesday
        assert arw.isoweekday() == 3

    def test_isocalendar(self):
        """Test isocalendar method."""
        arw = arrow.Arrow(2020, 5, 15)
        iso = arw.isocalendar()
        assert iso[0] == 2020  # year
        assert iso[1] > 0  # week

    def test_naive_property(self):
        """Test naive property."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        naive = arw.naive
        assert naive.tzinfo is None

    def test_date_property(self):
        """Test date property."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30)
        date = arw.date()
        assert date.year == 2020
        assert date.month == 5
        assert date.day == 15


class TestArrowComparisons:
    """Tests for Arrow comparison operations."""

    def test_equal(self):
        """Test equality."""
        arw1 = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        arw2 = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        assert arw1 == arw2

    def test_not_equal(self):
        """Test inequality."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arrow.Arrow(2020, 5, 16)
        assert arw1 != arw2

    def test_less_than(self):
        """Test less than."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arrow.Arrow(2020, 5, 16)
        assert arw1 < arw2

    def test_less_than_or_equal(self):
        """Test less than or equal."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arrow.Arrow(2020, 5, 15)
        assert arw1 <= arw2

    def test_greater_than(self):
        """Test greater than."""
        arw1 = arrow.Arrow(2020, 5, 16)
        arw2 = arrow.Arrow(2020, 5, 15)
        assert arw1 > arw2

    def test_greater_than_or_equal(self):
        """Test greater than or equal."""
        arw1 = arrow.Arrow(2020, 5, 16)
        arw2 = arrow.Arrow(2020, 5, 16)
        assert arw1 >= arw2


class TestArrowArithmetic:
    """Tests for Arrow arithmetic operations."""

    def test_add_timedelta(self):
        """Test adding timedelta."""
        arw = arrow.Arrow(2020, 5, 15)
        td = dt.timedelta(days=5)
        result = arw + td
        assert result.day == 20

    def test_subtract_timedelta(self):
        """Test subtracting timedelta."""
        arw = arrow.Arrow(2020, 5, 15)
        td = dt.timedelta(days=5)
        result = arw - td
        assert result.day == 10

    def test_subtract_arrow(self):
        """Test subtracting Arrow objects."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arrow.Arrow(2020, 5, 10)
        td = arw1 - arw2
        assert isinstance(td, dt.timedelta)
        assert td.days == 5
