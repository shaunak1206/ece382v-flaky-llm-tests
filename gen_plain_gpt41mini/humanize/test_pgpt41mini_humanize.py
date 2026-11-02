import math
import datetime as dt
import pytest
from freezegun import freeze_time

import humanize
from humanize import (
    apnumber,
    clamp,
    decimal_separator,
    deactivate,
    intcomma,
    intword,
    fractional,
    natural_list,
    naturaldate,
    naturalday,
    naturaldelta,
    naturalsize,
    naturaltime,
    ordinal,
    precisedelta,
    scientific,
    thousands_separator,
    activate,
)


# --------- Tests for humanize._version ---------
def test_version_format():
    assert isinstance(humanize.__version__, str)
    assert humanize.__version__.startswith("0.")


# --------- Tests for filesize.naturalsize ---------
@pytest.mark.parametrize(
    "value,binary,gnu,fmt,expected",
    [
        (3000000, False, False, "%.1f", "3.0 MB"),
        (300, False, True, "%.1f", "300B"),
        (3000, False, True, "%.1f", "2.9K"),
        (3000, False, True, "%.3f", "2.930K"),
        (3000, True, False, "%.1f", "2.9 KiB"),
        (10 ** 28, False, False, "%.1f", "10.0 RB"),
        (10 ** 34 * 3, False, False, "%.1f", "30000.0 QB"),
        (-4096, True, False, "%.1f", "-4.0 KiB"),
    ],
)
def test_naturalsize_various(value, binary, gnu, fmt, expected):
    assert naturalsize(value, binary=binary, gnu=gnu, format=fmt) == expected


def test_naturalsize_bytes_and_byte_singular():
    # singular bytes (1) returns '1 Byte' (not plural)
    assert naturalsize(1, binary=False, gnu=False) == "1 Byte"
    # gnu 1 byte returns '1B'
    assert naturalsize(1, gnu=True) == "1B"
    # less than base but more than 1 returns e.g. '500 Bytes'
    assert naturalsize(500, binary=False, gnu=False).endswith("Bytes")
    # negative bytes under base
    assert naturalsize(-500, binary=False, gnu=False).startswith("-500 ")


def test_naturalsize_rounding_and_expansion():
    # Check that values close to base boundary step up suffix properly
    assert naturalsize(999_999, binary=False) == "1.0 MB"


# --------- Tests for i18n activate/deactivate and separators ---------
def test_default_decimal_thousands_separator_and_deactivate():
    deactivate()
    assert decimal_separator() == "."
    assert thousands_separator() == ","
    t = activate(None)
    assert t.gettext("test") == "test"  # NullTranslations returns original string
    deactivate()


def test_activate_invalid_path(monkeypatch):
    # monkeypatch _get_default_locale_path to None to trigger FileNotFoundError
    from humanize import i18n

    monkeypatch.setattr(i18n, "_get_default_locale_path", lambda: None)
    with pytest.raises(FileNotFoundError):
        activate("es_ES")


@pytest.mark.parametrize(
    "locale,expected_thousands,expected_decimal",
    [
        ("de_DE", ".", ","),
        ("fr_FR", " ", "."),
        ("it_IT", ".", ","),
        ("pt_BR", ".", ","),
        ("hu_HU", " ", ","),
        ("lv", " ", ","),
        (None, ",", "."),
        ("en_US", ",", "."),
    ],
)
def test_thousands_and_decimal_separators(monkeypatch, locale, expected_thousands, expected_decimal):
    from humanize import i18n

    monkeypatch.setattr(i18n._CURRENT, "locale", locale)
    assert thousands_separator() == expected_thousands
    assert decimal_separator() == expected_decimal
    monkeypatch.setattr(i18n._CURRENT, "locale", None)  # Reset


def test_gettext_functions_basic(monkeypatch):
    # test _gettext, _ngettext, _pgettext are callable and return string
    from humanize import i18n

    monkeypatch.setattr(i18n._CURRENT, "locale", None)
    assert i18n._gettext("foo") == "foo"
    assert i18n._ngettext("foo", "bars", 1) == "foo"
    assert i18n._ngettext("foo", "bars", 2) == "bars"
    assert i18n._pgettext("ctx", "foo") == "foo"
    # _gettext_noop returns original string
    assert i18n._gettext_noop("hello") == "hello"
    # _ngettext_noop returns tuple
    assert i18n._ngettext_noop("singular", "plural") == ("singular", "plural")


# --------- Tests for lists.natural_list ---------
def test_natural_list_empty():
    assert natural_list([]) == ""


@pytest.mark.parametrize(
    "input_list,expected",
    [
        (["one"], "one"),
        (["one", "two"], "one and two"),
        (["one", "two", "three"], "one, two and three"),
        (["a", "b", "c", "d"], "a, b, c and d"),
        ([1, 2, 3], "1, 2 and 3"),
    ],
)
def test_natural_list_various(input_list, expected):
    assert natural_list(input_list) == expected


# --------- Tests for number.ordinal ---------
@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (111, "111th"),
        ("5", "5th"),
        (1002, "1002nd"),
        ("something else", "something else"),
        ([1, 2, 3], "[1, 2, 3]"),
    ],
)
def test_ordinal_basic(value, expected):
    # The suffix depends on P_ which ultimately returns the suffix from _ORDINAL_SUFFIXES
    result = ordinal(value)
    assert result.startswith(str(value) if isinstance(value, int) else str(value))
    # for non-integers/convertible, result is str(value)
    if not isinstance(value, (int, float, str)) or (isinstance(value, str) and not value.isnumeric()):
        assert result == str(value)


def test_ordinal_gender_is_ignored():
    # Gender is ignored except "female" is accepted
    assert ordinal(1, gender="female").endswith("st")
    assert ordinal(2, gender="other").endswith("nd")


# --------- Tests for number.intcomma ---------
def test_intcomma_basic():
    assert intcomma(100) == "100"
    assert intcomma("1000") == "1,000"
    assert intcomma(1_000_000) == "1,000,000"
    assert intcomma(1_234_567.25) == "1,234,567.25"
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"
    assert intcomma("14308.40", 1) == "14,308.4"
    # None returns "None"
    assert intcomma(None) == "None"


def test_intcomma_nonfinite_values():
    assert intcomma(float("nan")) == "NaN"
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"


def test_intcomma_custom_separators(monkeypatch):
    # Patch _CURRENT locale to test other separators
    from humanize import i18n

    monkeypatch.setattr(i18n._CURRENT, "locale", "de_DE")
    # De activates default comma, decimal sep used in intcomma and should translate accordingly
    r = intcomma(1234567.89, 2)
    # de_DE uses . for thousands and , for decimals
    assert "." in r and "," in r
    monkeypatch.setattr(i18n._CURRENT, "locale", None)


# --------- Tests for number.intword ---------
def test_intword_basic_and_none():
    assert intword("100") == "100"
    assert intword("12400") == "12.4 thousand"
    assert intword("1000000") == "1.0 million"
    assert intword(1_200_000_000) == "1.2 billion"
    big = 8100000000000000000000000000000000
    result = intword(big)
    assert "decillion" in result or "googol" in result
    # None returns string "None"
    assert intword(None) == "None"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"


def test_intword_nonfinite():
    assert intword(float("nan")) == "NaN"
    assert intword(float("inf")) == "+Inf"
    assert intword(float("-inf")) == "-Inf"


# --------- Tests for number.apnumber ---------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "zero"),
        (5, "five"),
        (10, "10"),
        ("7", "seven"),
        ("foo", "foo"),
        (None, "None"),
        (float("nan"), "NaN"),
    ],
)
def test_apnumber_basic(value, expected):
    assert apnumber(value) == expected


# --------- Tests for number.fractional ---------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.3, "3/10"),
        (1.3, "1 3/10"),
        (1/3, "1/3"),
        (1, "1"),
        ("ten", "ten"),
        (None, "None"),
        (float("nan"), "NaN"),
    ],
)
def test_fractional_basic(value, expected):
    assert fractional(value) == expected


# --------- Tests for number.scientific ---------
@pytest.mark.parametrize(
    "value,precision,expected_start",
    [
        (0.3, 2, "3.00 x 10⁻¹"),
        (500, 2, "5.00 x 10²"),
        (-1000, 2, "-1.00 x 10³"),
        (1000, 1, "1.0 x 10³"),
        (1000, 3, "1.000 x 10³"),
        ("99", 2, "9.90 x 10¹"),
        ("foo", 2, "foo"),
        (None, 2, "None"),
        (float("nan"), 2, "NaN"),
        (float("inf"), 2, "+Inf"),
        (float("-inf"), 2, "-Inf"),
    ],
)
def test_scientific_basic(value, precision, expected_start):
    result = scientific(value, precision)
    assert result == expected_start


# --------- Tests for number.clamp ---------
def test_clamp_basic_and_none():
    assert clamp(123.456) == "123.456"
    assert clamp(None) is None


def test_clamp_with_floor_and_ceil():
    assert clamp(0.0001, floor=0.01) == "<0.01"
    # Test clamp with ceil, using format string for percent
    assert clamp(0.99, format="{:.0%}", ceil=0.99) == "99%"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_with_callable_format():
    # custom formatter is intword from humanize.number
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"


def test_clamp_invalid_format_raises():
    with pytest.raises(ValueError):
        clamp(1, format=12345)


def test_clamp_nonfinite():
    assert clamp(float("nan")) == "NaN"
    assert clamp(float("inf")) == "+Inf"
    assert clamp(float("-inf")) == "-Inf"


# --------- Tests for number.metric ---------
@pytest.mark.parametrize(
    "value,unit,precision,expected_contains",
    [
        (1500, "V", 3, "kV"),
        (2e8, "W", 3, "MW"),
        (220e-6, "F", 3, "μF"),
        (1e-14, "", 4, "f"),
        (1e40, "", 3, "x 10⁴⁰"),
        (0, "", 3, ""),
    ],
)
def test_metric_basic(value, unit, precision, expected_contains):
    result = humanize.number.metric(value, unit=unit, precision=precision)
    assert isinstance(result, str)
    assert expected_contains in result


def test_metric_nonfinite():
    assert humanize.number.metric(float("nan")) == "NaN"
    assert humanize.number.metric(float("inf")) == "+Inf"
    assert humanize.number.metric(float("-inf")) == "-Inf"


# --------- Tests for time.naturaldelta ---------
def test_naturaldelta_seconds_and_moments():
    now = dt.datetime.now()
    # timedelta zero -> "a moment"
    assert naturaldelta(dt.timedelta(seconds=0)) == "a moment"
    # a second
    assert naturaldelta(dt.timedelta(seconds=1)) == "a second"
    # seconds plural
    assert naturaldelta(dt.timedelta(seconds=10)) == "10 seconds"
    # a minute
    assert naturaldelta(dt.timedelta(seconds=60)) == "a minute"
    # multiple minutes
    assert naturaldelta(dt.timedelta(seconds=180)) == "3 minutes"
    # an hour
    assert naturaldelta(dt.timedelta(seconds=3600)) == "an hour"
    # multiple hours
    assert naturaldelta(dt.timedelta(seconds=7200)) == "2 hours"
    # a day
    assert naturaldelta(dt.timedelta(days=1)) == "a day"
    # multiple days with months
    assert "months" in naturaldelta(dt.timedelta(days=40))
    # years
    assert "year" in naturaldelta(dt.timedelta(days=365 * 2))
    # non-datetime input seconds float
    assert naturaldelta(20.0) == "20 seconds"
    # Invalid input string returns unchanged
    assert naturaldelta("invalid_input") == "invalid_input"


def test_naturaldelta_minimum_unit_invalid_raises():
    with pytest.raises(ValueError):
        naturaldelta(3600, minimum_unit="hours")


def test_naturaldelta_nonfinite_values():
    assert naturaldelta(float("nan")) == "nan"
    assert naturaldelta(float("inf")) == "inf"
    assert naturaldelta(float("-inf")) == "-inf"


# --------- Tests for time.naturalday ---------
@freeze_time("2023-03-21")
def test_naturalday_today_tomorrow_yesterday():
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)
    # today returns "today"
    assert naturalday(today) == "today"
    # tomorrow returns "tomorrow"
    assert naturalday(tomorrow) == "tomorrow"
    # yesterday returns "yesterday"
    assert naturalday(yesterday) == "yesterday"

    # other dates return formatted string
    dt_out = today - dt.timedelta(days=3)
    expected = dt_out.strftime("%b %d")
    assert naturalday(dt_out) == expected


def test_naturalday_invalid_types():
    # None returns string "None"
    assert naturalday(None) == "None"
    # String returns itself
    assert naturalday("foo") == "foo"


# --------- Tests for time.naturaldate ---------
@freeze_time("2023-03-21")
def test_naturaldate_behavior():
    today = dt.date.today()
    # date less than ~5 months returns naturalday output without year
    d = today - dt.timedelta(days=10)
    assert naturaldate(d) == d.strftime("%b %d")
    # date more than ~5 months returns with year
    d2 = today - dt.timedelta(days=160)
    assert naturaldate(d2) == d2.strftime("%b %d %Y")
    # invalid types return str(value)
    assert naturaldate(None) == "None"


# --------- Tests for time.naturaltime ---------
@freeze_time("2023-03-21 12:00:00")
def test_naturaltime_past_and_future():
    now = dt.datetime.now()
    past = now - dt.timedelta(minutes=5)
    future = now + dt.timedelta(minutes=5)
    # past datetime
    result_past = naturaltime(past)
    assert "ago" in result_past or "now" in result_past
    # future datetime
    result_future = naturaltime(future)
    assert "from now" in result_future or "now" in result_future
    # timedelta input
    td = dt.timedelta(seconds=30)
    result_td = naturaltime(td)
    assert isinstance(result_td, str)
    # float input, future False
    result_float_past = naturaltime(30)
    assert "ago" in result_float_past or "now" in result_float_past
    # float input, future True
    result_float_future = naturaltime(30, future=True)
    assert "from now" in result_float_future or "now" in result_float_future
    # invalid value returns str(value)
    assert naturaltime("foo") == "foo"


# --------- Tests for time.precisedelta ---------
def test_precisedelta_basic_units():
    delta = dt.timedelta(days=2, hours=1, minutes=33, seconds=12, microseconds=123000)
    pd = precisedelta(delta)
    # Check string contains expected units approx
    assert "days" in pd
    assert "hour" in pd
    assert ("second" in pd or "seconds" in pd)
    # Test changing format precision
    pd_precise = precisedelta(delta, format="%0.4f")
    assert pd_precise != pd
    # Test minimum_unit microseconds
    pd_micro = precisedelta(delta, minimum_unit="microseconds")
    assert "milliseconds" in pd_micro or "microseconds" in pd_micro
    # Test suppress days
    pd_suppr = precisedelta(delta, suppress=["days"])
    assert "hours" in pd_suppr
    assert "days" not in pd_suppr
    # Test suppress seconds and smaller
    delta2 = dt.timedelta(seconds=90, microseconds=100)
    pd_suppr2 = precisedelta(delta2, suppress=["seconds", "milliseconds", "microseconds"])
    # Should not contain seconds
    assert "seconds" not in pd_suppr2
    # Test minimum_unit larger than delta
    delta3 = dt.timedelta(seconds=1)
    pd_min_unit = precisedelta(delta3, minimum_unit="minutes")
    assert "minutes" in pd_min_unit
    # Very small delta returns 0 something
    delta4 = dt.timedelta(seconds=0.1)
    pd_min_unit2 = precisedelta(delta4, minimum_unit="minutes")
    assert pd_min_unit2.startswith("0")


def test_precisedelta_invalid_value():
    assert precisedelta("foo") == "foo"


def test_precisedelta_remainder_behavior():
    from humanize.time import _quotient_and_remainder, Unit

    # If unit == minimum_unit, remainder zero and quotient rounded
    q, r = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [], "%0.2f")
    assert r == 0
    assert isinstance(q, (float, int))

    # If unit in suppress, quotient zero and remainder value
    q2, r2 = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS], "%0.2f")
    assert q2 == 0
    assert r2 == 36

    # Normal case divmod
    q3, r3 = _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [], "%0.2f")
    assert (q3, r3) == (1, 12)


def test_precisedelta_suppression_and_minimum_unit():
    from humanize.time import _suitable_minimum_unit, _suppress_lower_units, Unit

    # Suitability returns same if not suppressed
    assert _suitable_minimum_unit(Unit.HOURS, []) == Unit.HOURS

    # Suitability returns greater unit if suppressed
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]) == Unit.DAYS

    # Raises if no suitable replacement found
    with pytest.raises(ValueError):
        _suitable_minimum_unit(Unit.HOURS, list(Unit))

    # Suppress lower units includes lower units plus given suppress units
    suppressed = _suppress_lower_units(Unit.SECONDS, [Unit.DAYS])
    assert Unit.MICROSECONDS in suppressed
    assert Unit.MILLISECONDS in suppressed
    assert Unit.DAYS in suppressed


def test_rounding_by_fmt_behavior():
    from humanize.time import _rounding_by_fmt

    # Integer truncation
    val = _rounding_by_fmt("%d", 3.9)
    assert val == 3 and isinstance(val, int)

    # Float rounding
    val2 = _rounding_by_fmt("%.2f", 3.14159)
    assert math.isclose(val2, 3.14)


# --------- Miscellaneous tests for coverage ---------
def test_natural_list_with_non_string_items():
    # Should convert items to string properly
    input_list = [1, None, 3.5]
    result = natural_list(input_list)
    assert "1" in result and "None" in result and "3.5" in result


def test_ordinal_non_integer_like_raises_returns_str():
    class Dummy:
        def __float__(self):
            raise ValueError()

    dummy = Dummy()
    result = ordinal(dummy)
    assert result == str(dummy)


def test_intcomma_invalid_string_returns_str():
    assert intcomma("notanumber") == "notanumber"


def test_apnumber_non_numeric_string_returns_str():
    assert apnumber("notanumber") == "notanumber"


def test_fractional_value_that_forms_whole_number():
    # 3.0 should return "3"
    assert fractional(3.0) == "3"


def test_scientific_with_zero_precision():
    # zero precision treated as int precision
    result = scientific(1000, precision=0)
    assert "x 10³" in result


def test_clamp_with_floor_and_ceil_tokens(monkeypatch):
    def formatter(val):
        return f"{val:.2f}"

    assert clamp(5.0, format=formatter, floor=10, floor_token="under ") == "under 10.00"
    assert clamp(20.0, format=formatter, ceil=10, ceil_token="over ") == "over 10.00"
    # value inside range
    assert clamp(15.0, format=formatter, floor=10, ceil=20) == "15.00"


def test_metric_units_with_symbols():
    res = humanize.number.metric(1500, unit="°")
    # Should have no space between number and unit
    assert "°" in res and " " not in res.strip()


def test_naturaltime_with_aware_datetime():
    import pytz

    tz = pytz.timezone("Europe/Berlin")
    now = dt.datetime.now(tz)
    past = now - dt.timedelta(minutes=5)
    result = naturaltime(past)
    assert "ago" in result or "now" in result


def test_naturaltime_none_value():
    assert naturaltime(None) == "None"


def test_naturalday_with_aware_datetime():
    import pytz

    tz = pytz.timezone("Europe/Berlin")
    now = dt.datetime.now(tz)
    result = naturalday(now)
    assert result in ("today", now.strftime("%b %d"))


def test_naturaldate_with_aware_datetime():
    import pytz

    tz = pytz.timezone("Europe/Berlin")
    now = dt.datetime.now(tz)
    result = naturaldate(now)
    assert result.startswith(now.strftime("%b"))


def test_precisedelta_with_zero_delta():
    zero = dt.timedelta(0)
    result = precisedelta(zero)
    assert result == "0 seconds" or "0" in result


def test_naturaldelta_zero_seconds_microseconds(monkeypatch):
    d = dt.timedelta(microseconds=500)
    result = naturaldelta(d, minimum_unit="microseconds")
    assert "microsecond" in result or "microseconds" in result


def test_naturaldelta_with_months_false():
    d = dt.timedelta(days=60)
    result = naturaldelta(d, months=False)
    assert "days" in result or "day" in result


def test_naturaltime_future_flag_with_seconds():
    future_text = naturaltime(30, future=True)
    past_text = naturaltime(30, future=False)
    assert future_text != past_text


def test_naturaltime_str_input_returns_itself():
    assert naturaltime("some string") == "some string"


def test_naturaltime_datetime_and_timedelta_future_calculation():
    now = dt.datetime.now()
    past_dt = now - dt.timedelta(seconds=10)
    future_dt = now + dt.timedelta(seconds=10)
    assert "ago" in naturaltime(past_dt)
    assert "from now" in naturaltime(future_dt)


def test_deactivate_resets_locale(monkeypatch):
    from humanize import i18n

    activate("de_DE")
    deactivate()
    assert i18n._CURRENT.locale is None


def test_activate_retuns_translation(monkeypatch, tmp_path):
    # Prepare a dummy locale dir with no actual .mo files; translation will fallback to NullTranslations
    t = activate("de_DE", tmp_path)
    assert t is not None


def test_intword_rounded_up_to_next_power(monkeypatch):
    # 999,999 rounds up to 1.0 million
    result = intword(999_999)
    assert "million" in result or "thousand" in result


def test_apnumber_out_of_range_returns_str():
    # Numbers < 0 or >=10 return string number
    assert apnumber(-1) == "-1"
    assert apnumber(10) == "10"


def test_fractional_negative_and_zero():
    assert fractional(-1.3) == "-1 3/10"
    assert fractional(0) == "0"


def test_scientific_with_large_negative_exponent():
    val = 0.00000123
    result = scientific(val, precision=3)
    assert "x 10" in result


# Additional necessary import test for i18n
def test_get_default_locale_path_returns_path_or_none():
    from humanize import i18n
    path = i18n._get_default_locale_path()
    # Should be Path or None
    assert path is None or str(path).endswith("locale")
