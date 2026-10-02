"""Tests for humanize.number functions."""

import pytest

import humanize


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


# ---------------------------------------------------------------- ordinal


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0th"),
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (10, "10th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (101, "101st"),
        (111, "111th"),
        (112, "112th"),
        (113, "113th"),
        (1002, "1002nd"),
    ],
)
def test_ordinal_integers(value, expected):
    assert humanize.ordinal(value) == expected


def test_ordinal_other_inputs():
    assert humanize.ordinal("5") == "5th"
    assert humanize.ordinal(3.7) == "3rd"
    assert humanize.ordinal("something else") == "something else"
    assert humanize.ordinal(None) == "None"
    assert humanize.ordinal([1, 2, 3]) == "[1, 2, 3]"
    assert humanize.ordinal(float("inf")) == "+Inf"
    assert humanize.ordinal(float("-inf")) == "-Inf"
    assert humanize.ordinal(float("nan")) == "NaN"
    # Without a translation the gender does not change the English suffix.
    assert humanize.ordinal(1, gender="female") == "1st"
    assert humanize.ordinal(2, gender="female") == "2nd"


# ---------------------------------------------------------------- intcomma


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (100, "100"),
        (999, "999"),
        (1000, "1,000"),
        ("1000", "1,000"),
        (1_000_000, "1,000,000"),
        (-1_234_567, "-1,234,567"),
        (1_234_567.25, "1,234,567.25"),
        ("1,000", "1,000"),
    ],
)
def test_intcomma_basic(value, expected):
    assert humanize.intcomma(value) == expected


def test_intcomma_ndigits_and_invalid_values():
    assert humanize.intcomma(1234.5454545, 2) == "1,234.55"
    assert humanize.intcomma(14308.40, 1) == "14,308.4"
    assert humanize.intcomma("14308.40", 1) == "14,308.4"
    assert humanize.intcomma(1234.6, 0) == "1,235"
    assert humanize.intcomma(None) == "None"
    assert humanize.intcomma("foo") == "foo"
    assert humanize.intcomma(float("inf")) == "+Inf"
    assert humanize.intcomma(float("nan")) == "NaN"


# ---------------------------------------------------------------- intword


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (999, "999"),
        ("100", "100"),
        (1000, "1.0 thousand"),
        ("12400", "12.4 thousand"),
        (1_000_000, "1.0 million"),
        (1_200_000, "1.2 million"),
        (1_200_000_000, "1.2 billion"),
        (3 * 10**12, "3.0 trillion"),
        (8100000000000000000000000000000000, "8.1 decillion"),
        (10**100, "1.0 googol"),
        (-1_500_000, "-1.5 million"),
        (999_999, "1.0 million"),
    ],
)
def test_intword_values(value, expected):
    assert humanize.intword(value) == expected


def test_intword_formats_and_invalid_values():
    assert humanize.intword("1234000", "%0.3f") == "1.234 million"
    assert humanize.intword(1_234_567, "%.0f") == "1 million"
    assert humanize.intword(None) == "None"
    assert humanize.intword("not a number") == "not a number"
    assert humanize.intword(float("inf")) == "+Inf"


# ---------------------------------------------------------------- apnumber


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "zero"),
        (1, "one"),
        (5, "five"),
        (9, "nine"),
        (10, "10"),
        (100, "100"),
        (-1, "-1"),
        ("7", "seven"),
        (5.9, "five"),
        ("foo", "foo"),
        (None, "None"),
        (float("nan"), "NaN"),
    ],
)
def test_apnumber(value, expected):
    assert humanize.apnumber(value) == expected


# ---------------------------------------------------------------- fractional


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (float(1 / 3), "1/3"),
        (0.5, "1/2"),
        (2.5, "2 1/2"),
        (0.75, "3/4"),
        ("0.75", "3/4"),
        (1, "1"),
        (1.0, "1"),
        (0, "0"),
        (2.9999999, "3"),
        (-0.5, "-1/2"),
        (-1.5, "-1 1/2"),
        ("ten", "ten"),
        (None, "None"),
        (float("inf"), "+Inf"),
        (float("nan"), "NaN"),
    ],
)
def test_fractional(value, expected):
    assert humanize.fractional(value) == expected


# ---------------------------------------------------------------- scientific


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.3, "3.00 x 10⁻¹"),
        (500, "5.00 x 10²"),
        (-1000, "-1.00 x 10³"),
        ("99", "9.90 x 10¹"),
        (0, "0.00 x 10⁰"),
        (123456, "1.23 x 10⁵"),
        (0.00001, "1.00 x 10⁻⁵"),
    ],
)
def test_scientific_default_precision(value, expected):
    assert humanize.scientific(value) == expected


def test_scientific_precision_and_invalid_values():
    assert humanize.scientific(1000, 1) == "1.0 x 10³"
    assert humanize.scientific(1000, 3) == "1.000 x 10³"
    assert humanize.scientific(1000, 0) == "1 x 10³"
    assert humanize.scientific("foo") == "foo"
    assert humanize.scientific(None) == "None"
    assert humanize.scientific(float("-inf")) == "-Inf"


# ---------------------------------------------------------------- clamp


def test_clamp_limits_and_tokens():
    assert humanize.clamp(123.456) == "123.456"
    assert humanize.clamp(0.0001, floor=0.01) == "<0.01"
    assert humanize.clamp(50, ceil=10) == ">10"
    assert humanize.clamp(5, floor=1, ceil=10) == "5"
    assert humanize.clamp(5, floor=10) == "<10"
    assert humanize.clamp(0.0001, floor=0.01, floor_token="under ") == "under 0.01"
    assert humanize.clamp(500, ceil=100, ceil_token="over ") == "over 100"


def test_clamp_formats():
    assert humanize.clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert humanize.clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"
    assert humanize.clamp(1, format=humanize.intword, floor=1e6, floor_token="under ") == (
        "under 1.0 million"
    )
    assert humanize.clamp(2500, format=lambda v: f"<{v}>") == "<2500>"


def test_clamp_none_and_non_finite():
    assert humanize.clamp(None) is None
    assert humanize.clamp(float("inf")) == "+Inf"
    assert humanize.clamp(float("nan")) == "NaN"


def test_clamp_invalid_format_raises():
    with pytest.raises(ValueError):
        humanize.clamp(5, format=123)


# ---------------------------------------------------------------- metric


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (1500, "V", "1.50 kV"),
        (2e8, "W", "200 MW"),
        (220e-6, "F", "220 μF"),
        (1000, "", "1.00 k"),
        (123000, "", "123 k"),
        (-1500, "V", "-1.50 kV"),
        (0, "", "0.00"),
        (1, "", "1.00"),
        (0.001, "", "1.00 m"),
        (90, "°", "90.0°"),
        (999_999, "", "1.00 M"),
    ],
)
def test_metric_values(value, unit, expected):
    assert humanize.metric(value, unit) == expected


def test_metric_precision_and_extremes():
    assert humanize.metric(1e-14, precision=4) == "10.00 f"
    assert humanize.metric(1234, precision=4) == "1.234 k"
    assert humanize.metric(1e40) == "1.00 x 10⁴⁰"


def test_metric_non_finite():
    assert humanize.metric(float("inf")) == "+Inf"
    assert humanize.metric(float("nan")) == "NaN"
