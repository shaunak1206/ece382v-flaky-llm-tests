"""Unit tests for humanize.number: ordinal, intcomma, intword, apnumber,
fractional, scientific, clamp, and metric.

Based only on the public function signatures and docstrings in
subjects/humanize/src/humanize/number.py.
"""

import pytest

from humanize import number


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (101, "101st"),
        (103, "103rd"),
        (111, "111th"),
        (1002, "1002nd"),
    ],
)
def test_ordinal_basic(value, expected):
    assert number.ordinal(value) == expected


def test_ordinal_non_numeric_string_returned_unchanged():
    assert number.ordinal("something else") == "something else"


def test_ordinal_gender_has_no_effect_without_translation():
    # Without an active translation, pgettext falls back to the English
    # suffix regardless of the requested gender.
    assert number.ordinal(1, gender="female") == number.ordinal(1, gender="male")


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_ordinal_non_finite_values(value, expected):
    assert number.ordinal(value) == expected


@pytest.mark.parametrize(
    "value, kwargs, expected",
    [
        (100, {}, "100"),
        ("1000", {}, "1,000"),
        (1_000_000, {}, "1,000,000"),
        (1_234_567.25, {}, "1,234,567.25"),
        (1234.5454545, {"ndigits": 2}, "1,234.55"),
        (14308.40, {"ndigits": 1}, "14,308.4"),
        ("14308.40", {"ndigits": 1}, "14,308.4"),
    ],
)
def test_intcomma_basic(value, kwargs, expected):
    assert number.intcomma(value, **kwargs) == expected


def test_intcomma_none_returns_string_none():
    assert number.intcomma(None) == "None"


def test_intcomma_negative_float():
    assert number.intcomma(-1234567.25) == "-1,234,567.25"


@pytest.mark.parametrize(
    "value, format_, expected",
    [
        ("100", "%.1f", "100"),
        ("12400", "%.1f", "12.4 thousand"),
        ("1000000", "%.1f", "1.0 million"),
        (1_200_000_000, "%.1f", "1.2 billion"),
        (8100000000000000000000000000000000, "%.1f", "8.1 decillion"),
        ("1234000", "%0.3f", "1.234 million"),
    ],
)
def test_intword_basic(value, format_, expected):
    assert number.intword(value, format_) == expected


def test_intword_negative_value():
    assert number.intword(-1_200_000) == "-1.2 million"


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "zero"),
        (1, "one"),
        (2, "two"),
        (3, "three"),
        (4, "four"),
        (5, "five"),
        (6, "six"),
        (7, "seven"),
        (8, "eight"),
        (9, "nine"),
    ],
)
def test_apnumber_spells_out_single_digits(value, expected):
    assert number.apnumber(value) == expected


def test_apnumber_ge_ten_is_left_as_digits():
    assert number.apnumber(10) == "10"


def test_apnumber_none_returns_string_none():
    assert number.apnumber(None) == "None"


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (1 / 3, "1/3"),
        (1, "1"),
    ],
)
def test_fractional_basic(value, expected):
    assert number.fractional(value) == expected


def test_fractional_negative_mixed_fraction():
    assert number.fractional(-1.3) == "-1 3/10"


@pytest.mark.parametrize(
    "value, precision, expected",
    [
        (0.3, 2, "3.00 x 10⁻¹"),
        (500, 2, "5.00 x 10²"),
        (-1000, 2, "-1.00 x 10³"),
        (1000, 1, "1.0 x 10³"),
        (1000, 3, "1.000 x 10³"),
        ("99", 2, "9.90 x 10¹"),
    ],
)
def test_scientific_basic(value, precision, expected):
    assert number.scientific(value, precision) == expected


def test_clamp_below_floor_uses_floor_token():
    assert number.clamp(0.0001, floor=0.01) == "<0.01"


def test_clamp_with_callable_format_and_custom_floor_token():
    result = number.clamp(1, format=number.intword, floor=1e6, floor_token="under ")
    assert result == "under 1.0 million"


def test_clamp_invalid_format_raises_value_error():
    with pytest.raises(ValueError):
        number.clamp(1, format=12345)


@pytest.mark.parametrize(
    "value, unit, kwargs, expected",
    [
        (1500, "V", {}, "1.50 kV"),
        (2e8, "W", {}, "200 MW"),
        (220e-6, "F", {}, "220 μF"),
        (1e-14, "", {"precision": 4}, "10.00 f"),
    ],
)
def test_metric_basic(value, unit, kwargs, expected):
    assert number.metric(value, unit, **kwargs) == expected


def test_metric_too_large_falls_back_to_scientific():
    assert number.metric(1e40) == "1.00 x 10⁴⁰"
