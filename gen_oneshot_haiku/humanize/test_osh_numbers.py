"""Tests for humanize.number module."""

import pytest
from humanize import ordinal, intcomma, intword, apnumber, fractional, scientific, clamp, metric


class TestOrdinal:
    """Tests for ordinal function."""

    def test_ordinal_basic(self):
        assert ordinal(1) == "1st"
        assert ordinal(2) == "2nd"
        assert ordinal(3) == "3rd"
        assert ordinal(4) == "4th"
        assert ordinal(5) == "5th"

    def test_ordinal_teens(self):
        assert ordinal(11) == "11th"
        assert ordinal(12) == "12th"
        assert ordinal(13) == "13th"

    def test_ordinal_tens(self):
        assert ordinal(21) == "21st"
        assert ordinal(22) == "22nd"
        assert ordinal(23) == "23rd"
        assert ordinal(24) == "24th"
        assert ordinal(101) == "101st"
        assert ordinal(111) == "111th"
        assert ordinal(121) == "121st"

    def test_ordinal_large(self):
        assert ordinal(1000) == "1000th"
        assert ordinal(1002) == "1002nd"
        assert ordinal(103) == "103rd"

    def test_ordinal_string(self):
        assert ordinal("1") == "1st"
        assert ordinal("2") == "2nd"

    def test_ordinal_invalid(self):
        assert ordinal("something") == "something"
        assert ordinal([1, 2, 3]) == "[1, 2, 3]"


class TestIntcomma:
    """Tests for intcomma function."""

    def test_intcomma_basic(self):
        assert intcomma(100) == "100"
        assert intcomma(1000) == "1,000"
        assert intcomma(1000000) == "1,000,000"

    def test_intcomma_string_input(self):
        assert intcomma("1000") == "1,000"
        assert intcomma("1000000") == "1,000,000"

    def test_intcomma_floats(self):
        assert intcomma(1234567.25) == "1,234,567.25"
        assert intcomma(1234.5) == "1,234.5"

    def test_intcomma_with_ndigits(self):
        assert intcomma(1234.5454545, 2) == "1,234.55"
        assert intcomma(14308.40, 1) == "14,308.4"

    def test_intcomma_negative(self):
        assert intcomma(-1000) == "-1,000"

    def test_intcomma_none(self):
        assert intcomma(None) == "None"

    def test_intcomma_small(self):
        assert intcomma(99) == "99"


class TestIntword:
    """Tests for intword function."""

    def test_intword_small(self):
        assert intword(100) == "100"
        assert intword("100") == "100"

    def test_intword_thousands(self):
        assert intword("12400") == "12.4 thousand"
        assert intword(100000) == "100.0 thousand"

    def test_intword_millions(self):
        assert intword("1000000") == "1.0 million"
        assert intword(1200000) == "1.2 million"
        assert intword(1200000000) == "1.2 billion"

    def test_intword_custom_format(self):
        assert intword("1234000", "%0.3f") == "1.234 million"

    def test_intword_negative(self):
        result = intword(-1200000)
        assert "1.2 million" in result
        assert "-" in result

    def test_intword_none(self):
        assert intword(None) == "None"

    def test_intword_large(self):
        result = intword(8100000000000000000000000000000000)
        assert "decillion" in result


class TestApnumber:
    """Tests for apnumber function."""

    def test_apnumber_zero_to_nine(self):
        assert apnumber(0) == "zero"
        assert apnumber(1) == "one"
        assert apnumber(2) == "two"
        assert apnumber(3) == "three"
        assert apnumber(4) == "four"
        assert apnumber(5) == "five"
        assert apnumber(6) == "six"
        assert apnumber(7) == "seven"
        assert apnumber(8) == "eight"
        assert apnumber(9) == "nine"

    def test_apnumber_ten_plus(self):
        assert apnumber(10) == "10"
        assert apnumber(11) == "11"
        assert apnumber(100) == "100"

    def test_apnumber_string(self):
        assert apnumber("7") == "seven"
        assert apnumber("10") == "10"

    def test_apnumber_invalid(self):
        assert apnumber("foo") == "foo"

    def test_apnumber_none(self):
        assert apnumber(None) == "None"


class TestFractional:
    """Tests for fractional function."""

    def test_fractional_simple(self):
        assert fractional(0.3) == "3/10"
        assert fractional(1) == "1"

    def test_fractional_mixed(self):
        assert fractional(1.3) == "1 3/10"

    def test_fractional_third(self):
        result = fractional(float(1/3))
        assert "1/3" in result

    def test_fractional_invalid(self):
        assert fractional("ten") == "ten"

    def test_fractional_none(self):
        assert fractional(None) == "None"

    def test_fractional_zero(self):
        assert fractional(0) == "0"

    def test_fractional_negative(self):
        result = fractional(-1.3)
        assert "-" in result or "1" in result


class TestScientific:
    """Tests for scientific function."""

    def test_scientific_basic(self):
        result = scientific(500)
        assert "5.00 x 10" in result

    def test_scientific_decimal(self):
        result = scientific(float(0.3))
        assert "x 10" in result

    def test_scientific_negative(self):
        result = scientific(-1000)
        assert "-" in result
        assert "x 10" in result

    def test_scientific_precision(self):
        result1 = scientific(1000, 1)
        result2 = scientific(1000, 3)
        assert "1.0" in result1
        assert "1.000" in result2

    def test_scientific_string(self):
        result = scientific("99")
        assert "x 10" in result

    def test_scientific_invalid(self):
        assert scientific("foo") == "foo"

    def test_scientific_none(self):
        assert scientific(None) == "None"


class TestClamp:
    """Tests for clamp function."""

    def test_clamp_no_limits(self):
        assert clamp(123.456) == "123.456"

    def test_clamp_floor(self):
        assert clamp(0.0001, floor=0.01) == "<0.01"

    def test_clamp_ceil(self):
        result = clamp(0.999, format="{:.0%}", ceil=0.99)
        assert "%" in result

    def test_clamp_with_callable(self):
        result = clamp(1, format=intword, floor=1e6, floor_token="under ")
        assert "under" in result

    def test_clamp_none(self):
        assert clamp(None) is None

    def test_clamp_with_custom_tokens(self):
        result = clamp(5, floor=10, floor_token="MIN: ")
        assert result == "MIN: 10"


class TestMetric:
    """Tests for metric function."""

    def test_metric_basic(self):
        result = metric(1500, "V")
        assert "k" in result or "1.5" in result

    def test_metric_large(self):
        result = metric(2e8, "W")
        assert "M" in result

    def test_metric_small(self):
        result = metric(220e-6, "F")
        assert "μ" in result or "u" in result

    def test_metric_no_unit(self):
        result = metric(1000)
        assert "1" in result

    def test_metric_precision(self):
        result = metric(1e-14, precision=4)
        assert "f" in result or "10" in result
