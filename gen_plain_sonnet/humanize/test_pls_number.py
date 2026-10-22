"""Unit tests for humanize.number functions."""

import pytest

import humanize
from humanize import (
    apnumber,
    clamp,
    fractional,
    intcomma,
    intword,
    metric,
    ordinal,
    scientific,
)


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


# ordinal ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0th"),
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (9, "9th"),
        (10, "10th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (24, "24th"),
        (101, "101st"),
        (1002, "1002nd"),
        (103, "103rd"),
    ],
)
def test_ordinal_basic_suffixes(value, expected):
    assert ordinal(value) == expected


@pytest.mark.parametrize("value", [11, 12, 13, 111, 112, 113, 1011, 1012, 1013])
def test_ordinal_teens_use_th(value):
    assert ordinal(value) == f"{value}th"


def test_ordinal_accepts_numeric_strings_and_floats():
    assert ordinal("5") == "5th"
    assert ordinal("22") == "22nd"
    assert ordinal(3.9) == "3rd"
    assert ordinal(2, gender="female") == "2nd"


def test_ordinal_invalid_and_not_finite():
    assert ordinal("something else") == "something else"
    assert ordinal(None) == "None"
    assert ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert ordinal(float("nan")) == "NaN"
    assert ordinal(float("inf")) == "+Inf"
    assert ordinal(float("-inf")) == "-Inf"


# intcomma --------------------------------------------------------------


def test_intcomma_integers_including_negative():
    assert intcomma(0) == "0"
    assert intcomma(100) == "100"
    assert intcomma(999) == "999"
    assert intcomma(1000) == "1,000"
    assert intcomma(45000) == "45,000"
    assert intcomma(1_000_000) == "1,000,000"
    assert intcomma(-999) == "-999"
    assert intcomma(-1234567) == "-1,234,567"


def test_intcomma_strings_and_floats():
    assert intcomma("1000") == "1,000"
    assert intcomma("1,234,567") == "1,234,567"
    assert intcomma("1234567.25") == "1,234,567.25"
    assert intcomma(1_234_567.25) == "1,234,567.25"
    assert intcomma(1234.5) == "1,234.5"


def test_intcomma_ndigits_rounds():
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"
    assert intcomma("14308.40", 1) == "14,308.4"
    assert intcomma(1234, 2) == "1,234.00"
    assert intcomma(1234.5, 0) == "1,234"


def test_intcomma_invalid_input_returns_str():
    assert intcomma(None) == "None"
    assert intcomma("abc") == "abc"
    assert intcomma([1]) == "[1]"


def test_intcomma_not_finite():
    assert intcomma(float("nan")) == "NaN"
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"
    assert intcomma("inf") == "+Inf"


# intword ---------------------------------------------------------------


def test_intword_below_thousand_is_plain():
    assert intword(0) == "0"
    assert intword(100) == "100"
    assert intword("100") == "100"
    assert intword(999) == "999"


@pytest.mark.parametrize(
    "value, expected",
    [
        (1000, "1.0 thousand"),
        ("12400", "12.4 thousand"),
        (1_000_000, "1.0 million"),
        (12_345_678, "12.3 million"),
        (1_200_000_000, "1.2 billion"),
        (10**12, "1.0 trillion"),
        (10**15, "1.0 quadrillion"),
        (10**18, "1.0 quintillion"),
        (10**21, "1.0 sextillion"),
        (10**24, "1.0 septillion"),
        (10**27, "1.0 octillion"),
        (10**30, "1.0 nonillion"),
        (10**33, "1.0 decillion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (10**100, "1.0 googol"),
        (10**101, "10.0 googol"),
    ],
)
def test_intword_scales(value, expected):
    assert intword(value) == expected


def test_intword_rounding_and_negative_values():
    assert intword(999_999) == "1.0 million"
    assert intword(999_999_999) == "1.0 billion"
    assert intword(-1_200_000) == "-1.2 million"
    assert intword(-500) == "-500"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"
    assert intword(1_500_000, "%.0f") == "2 million"
    assert intword(1_500_000, "%.2f") == "1.50 million"


def test_intword_invalid_and_not_finite():
    assert intword(None) == "None"
    assert intword("abc") == "abc"
    assert intword(float("inf")) == "+Inf"
    assert intword(float("-inf")) == "-Inf"
    assert intword(float("nan")) == "NaN"


# apnumber --------------------------------------------------------------


def test_apnumber_spells_out_zero_to_nine():
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for number, word in enumerate(words):
        assert apnumber(number) == word
    assert apnumber("7") == "seven"


def test_apnumber_outside_range_and_invalid():
    assert apnumber(10) == "10"
    assert apnumber(100) == "100"
    assert apnumber(-1) == "-1"
    assert apnumber("foo") == "foo"
    assert apnumber(None) == "None"
    assert apnumber(float("nan")) == "NaN"
    assert apnumber(float("inf")) == "+Inf"


# fractional ------------------------------------------------------------


def test_fractional_fractions_and_mixed_numbers():
    assert fractional(0.3) == "3/10"
    assert fractional(0.5) == "1/2"
    assert fractional(0.25) == "1/4"
    assert fractional(float(1 / 3)) == "1/3"
    assert fractional(1.3) == "1 3/10"
    assert fractional(2.5) == "2 1/2"
    assert fractional("1.5") == "1 1/2"


def test_fractional_whole_and_negative_numbers():
    assert fractional(0) == "0"
    assert fractional(1) == "1"
    assert fractional(1.0) == "1"
    assert fractional(2.9999999) == "3"
    assert fractional(-0.5) == "-1/2"
    assert fractional(-1.3) == "-1 3/10"


def test_fractional_invalid_and_not_finite():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"
    assert fractional(float("inf")) == "+Inf"
    assert fractional(float("nan")) == "NaN"


# scientific ------------------------------------------------------------


def test_scientific_basic_and_zero_exponent():
    assert scientific(0.3) == "3.00 x 10⁻¹"
    assert scientific(500) == "5.00 x 10²"
    assert scientific(-1000) == "-1.00 x 10³"
    assert scientific("99") == "9.90 x 10¹"
    assert scientific(12345) == "1.23 x 10⁴"
    assert scientific(0) == "0.00 x 10⁰"
    assert scientific(1) == "1.00 x 10⁰"


def test_scientific_precision_and_multi_digit_exponents():
    assert scientific(1000, 1) == "1.0 x 10³"
    assert scientific(1000, 3) == "1.000 x 10³"
    assert scientific(1000, 0) == "1 x 10³"
    assert scientific(1e-10) == "1.00 x 10⁻¹⁰"
    assert scientific(1e100) == "1.00 x 10¹⁰⁰"


def test_scientific_invalid_and_not_finite():
    assert scientific("foo") == "foo"
    assert scientific(None) == "None"
    assert scientific(float("inf")) == "+Inf"
    assert scientific(float("-inf")) == "-Inf"
    assert scientific(float("nan")) == "NaN"


# clamp -----------------------------------------------------------------


def test_clamp_without_limits_formats_value():
    assert clamp(123.456) == "123.456"
    assert clamp(5, floor=1, ceil=10) == "5"


def test_clamp_floor_and_ceil_tokens():
    assert clamp(0.0001, floor=0.01) == "<0.01"
    assert clamp(5, floor=10) == "<10"
    assert clamp(10, floor=10) == "10"
    assert clamp(0, floor=1, floor_token="under ") == "under 1"
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"
    assert clamp(100, ceil=10, ceil_token="over ") == "over 10"


def test_clamp_with_callable_format():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"
    assert clamp(5_000_000, format=intword) == "5.0 million"


def test_clamp_none_and_not_finite():
    assert clamp(None) is None
    assert clamp(float("nan")) == "NaN"
    assert clamp(float("inf")) == "+Inf"
    assert clamp(float("-inf")) == "-Inf"


def test_clamp_invalid_format_raises():
    with pytest.raises(ValueError):
        clamp(5, format=123)


# metric ----------------------------------------------------------------


def test_metric_large_and_small_prefixes():
    assert metric(1500, "V") == "1.50 kV"
    assert metric(2e8, "W") == "200 MW"
    assert metric(123456) == "123 k"
    assert metric(1234, precision=2) == "1.2 k"
    assert metric(220e-6, "F") == "220 μF"
    assert metric(0.5, "A") == "500 mA"
    assert metric(1e-14, precision=4) == "10.00 f"


def test_metric_unprefixed_negative_and_degree_values():
    assert metric(1) == "1.00"
    assert metric(0) == "0.00"
    assert metric(1.5, "m") == "1.50 m"
    assert metric(-1500, "V") == "-1.50 kV"
    assert metric(45, "°") == "45.0°"


def test_metric_rounding_carries_to_next_prefix():
    assert metric(999999) == "1.00 M"


def test_metric_extreme_values_use_scientific():
    assert metric(1e40) == "1.00 x 10⁴⁰"
    assert metric(1e40, "W") == "1.00 x 10⁴⁰W"


def test_metric_not_finite():
    assert metric(float("inf")) == "+Inf"
    assert metric(float("-inf")) == "-Inf"
    assert metric(float("nan")) == "NaN"
