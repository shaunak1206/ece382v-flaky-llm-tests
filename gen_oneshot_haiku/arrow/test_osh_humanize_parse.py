"""Tests for Arrow humanize, dehumanize, and parsing methods."""

import datetime as dt
import pytest
import arrow
from freezegun import freeze_time


class TestArrowHumanize:
    """Tests for Arrow.humanize() method."""

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_now(self):
        """Test humanize with current time."""
        arw = arrow.now()
        result = arw.humanize()
        assert "now" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_past(self):
        """Test humanize with past time."""
        past = arrow.Arrow(2023, 1, 15, 10, 0)
        result = past.humanize()
        assert "ago" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_future(self):
        """Test humanize with future time."""
        future = arrow.Arrow(2023, 1, 15, 14, 0)
        result = future.humanize()
        assert "from now" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_with_other_arrow(self):
        """Test humanize with other Arrow object."""
        arw1 = arrow.Arrow(2023, 1, 15, 12, 0)
        arw2 = arrow.Arrow(2023, 1, 15, 10, 0)
        result = arw1.humanize(arw2)
        assert "2 hours" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_with_other_datetime(self):
        """Test humanize with datetime object."""
        arw = arrow.Arrow(2023, 1, 15, 12, 0)
        other = dt.datetime(2023, 1, 15, 10, 0)
        result = arw.humanize(other)
        assert "2 hours" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_only_distance(self):
        """Test humanize with only_distance."""
        past = arrow.Arrow(2023, 1, 15, 10, 0)
        result = past.humanize(only_distance=True)
        assert "2 hours" in result
        assert "ago" not in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_granularity_hour(self):
        """Test humanize with hour granularity."""
        past = arrow.Arrow(2023, 1, 15, 11, 45)
        result = past.humanize(granularity="hour")
        assert result is not None

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_seconds_ago(self):
        """Test humanize for seconds ago."""
        past = arrow.Arrow(2023, 1, 15, 11, 59, 45)
        result = past.humanize()
        assert "second" in result or "15 second" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_minute_ago(self):
        """Test humanize for minute ago."""
        past = arrow.Arrow(2023, 1, 15, 11, 59)
        result = past.humanize()
        assert "minute" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_day_ago(self):
        """Test humanize for day ago."""
        past = arrow.Arrow(2023, 1, 14, 12, 0)
        result = past.humanize()
        assert "day" in result

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_locale_fr(self):
        """Test humanize with French locale."""
        past = arrow.Arrow(2023, 1, 15, 10, 0)
        result = past.humanize(locale="fr")
        assert result is not None

    @freeze_time("2023-01-15 12:00:00")
    def test_humanize_locale_de(self):
        """Test humanize with German locale."""
        past = arrow.Arrow(2023, 1, 15, 10, 0)
        result = past.humanize(locale="de")
        assert result is not None


class TestArrowParsing:
    """Tests for Arrow parsing and string parsing."""

    def test_parse_iso_date(self):
        """Test parsing ISO date."""
        arw = arrow.get("2020-05-15")
        assert arw.year == 2020
        assert arw.month == 5
        assert arw.day == 15

    def test_parse_iso_datetime(self):
        """Test parsing ISO datetime."""
        arw = arrow.get("2020-05-15T10:30:45Z")
        assert arw.year == 2020
        assert arw.hour == 10

    def test_parse_iso_datetime_with_tz(self):
        """Test parsing ISO datetime with timezone."""
        arw = arrow.get("2020-05-15T10:30:45+05:00")
        assert arw.year == 2020

    def test_parse_us_date(self):
        """Test parsing US date format."""
        arw = arrow.get("05/15/2020", "MM/DD/YYYY")
        assert arw.year == 2020
        assert arw.month == 5
        assert arw.day == 15

    def test_parse_european_date(self):
        """Test parsing European date format."""
        arw = arrow.get("15-05-2020", "DD-MM-YYYY")
        assert arw.year == 2020
        assert arw.month == 5
        assert arw.day == 15

    def test_parse_with_multiple_formats(self):
        """Test parsing with multiple possible formats."""
        arw = arrow.get("2020-05-15", ["YYYY-MM-DD", "DD-MM-YYYY"])
        assert arw.year == 2020

    def test_parse_rfc2822(self):
        """Test parsing RFC2822 format."""
        arw = arrow.get("Wed, 15 May 2020 10:30:45 +0000", "ddd, DD MMM YYYY HH:mm:ss ZZ")
        assert arw.year == 2020

    def test_parse_invalid_date_raises(self):
        """Test that parsing invalid date raises."""
        with pytest.raises(Exception):
            arrow.get("not a date")

    def test_parse_timestamp_direct(self):
        """Test parsing timestamp directly."""
        ts = 1589462745
        arw = arrow.get(ts)
        assert isinstance(arw, arrow.Arrow)


class TestArrowRepresentations:
    """Tests for Arrow string representations."""

    def test_str_representation(self):
        """Test string representation."""
        arw = arrow.Arrow(2020, 5, 15, 10, 30, 45)
        result = str(arw)
        assert "2020" in result

    def test_repr_representation(self):
        """Test repr representation."""
        arw = arrow.Arrow(2020, 5, 15)
        result = repr(arw)
        assert "Arrow" in result
        assert "2020" in result

    def test_format_representation(self):
        """Test format representation."""
        arw = arrow.Arrow(2020, 5, 15)
        result = f"{arw}"
        assert "2020" in result

    def test_format_custom(self):
        """Test format with format string."""
        arw = arrow.Arrow(2020, 5, 15)
        result = f"{arw:YYYY-MM-DD}"
        assert result == "2020-05-15"


class TestArrowAdvanced:
    """Tests for advanced Arrow features."""

    def test_ambiguous_property(self):
        """Test ambiguous property for DST transitions."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.ambiguous
        assert isinstance(result, bool)

    def test_imaginary_property(self):
        """Test imaginary property for DST transitions."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.imaginary
        assert isinstance(result, bool)

    def test_tzname(self):
        """Test tzname method."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        result = arw.tzname()
        assert result is not None

    def test_dst_property(self):
        """Test dst property."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="US/Eastern")
        result = arw.dst()
        assert result is None or isinstance(result, dt.timedelta)

    def test_utcoffset_property(self):
        """Test utcoffset method."""
        arw = arrow.Arrow(2020, 5, 15, tzinfo="UTC")
        result = arw.utcoffset()
        assert result is None or isinstance(result, dt.timedelta)

    def test_quarter_property(self):
        """Test quarter property."""
        arw1 = arrow.Arrow(2020, 1, 15)
        arw2 = arrow.Arrow(2020, 4, 15)
        arw3 = arrow.Arrow(2020, 7, 15)
        arw4 = arrow.Arrow(2020, 10, 15)
        assert arw1.quarter == 1
        assert arw2.quarter == 2
        assert arw3.quarter == 3
        assert arw4.quarter == 4

    def test_week_property(self):
        """Test week property."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.week
        assert isinstance(result, int)
        assert 1 <= result <= 53

    def test_hash(self):
        """Test that Arrow objects are hashable."""
        arw1 = arrow.Arrow(2020, 5, 15)
        arw2 = arrow.Arrow(2020, 5, 15)
        # Both should have the same hash
        assert hash(arw1) == hash(arw2)

    def test_year_property_is_int(self):
        """Test that year property returns int."""
        arw = arrow.Arrow(2020, 5, 15)
        assert isinstance(arw.year, int)


class TestArrowEdgeCases:
    """Tests for Arrow edge cases."""

    def test_leap_year_date(self):
        """Test handling of leap year dates."""
        arw = arrow.Arrow(2020, 2, 29)
        assert arw.day == 29

    def test_end_of_month_shift(self):
        """Test shifting to end of month."""
        arw = arrow.Arrow(2020, 1, 31)
        result = arw.shift(months=1)
        assert result.month == 2

    def test_negative_shift(self):
        """Test negative shifts."""
        arw = arrow.Arrow(2020, 5, 15)
        result = arw.shift(days=-5)
        assert result.day == 10

    def test_max_min_timestamps(self):
        """Test Arrow with extreme timestamps."""
        # Should not raise for reasonable timestamps
        ts = 1000000000
        arw = arrow.get(ts)
        assert isinstance(arw, arrow.Arrow)

    def test_year_boundaries(self):
        """Test year boundary operations."""
        arw = arrow.Arrow(2020, 12, 31, 23, 59, 59)
        result = arw.shift(seconds=1)
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1
