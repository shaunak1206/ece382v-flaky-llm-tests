import datetime as dt
import gettext
from pathlib import Path

import pytest
from freezegun import freeze_time

import humanize
from humanize import (
    activate,
    apnumber,
    clamp,
    deactivate,
    decimal_separator,
    fractional,
    intcomma,
    intword,
    metric,
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
)
from humanize import i18n
from humanize.filesize import suffixes
from humanize.number import _format_not_finite
from humanize.time import Unit, _rounding_by_fmt, _suitable_minimum_unit, _suppress_lower_units


def test_version_is_exposed():
    assert humanize.__version__ == "0.1.dev1"


def test_all_exports_include_expected_names():
    exported = set(humanize.__all__)
    for name in {
        "__version__",
        "activate",
        "apnumber",
        "clamp",
        "deactivate",
        "decimal_separator",
        "fractional",
        "intcomma",
        "intword",
        "metric",
        "natural_list",
        "naturaldate",
        "naturalday",
        "naturaldelta",
        "naturalsize",
        "naturaltime",
        "ordinal",
        "precisedelta",
        "scientific",
        "thousands_separator",
    }:
        assert name in exported


def test_natural_list_empty():
    assert natural_list([]) == ""


def test_natural_list_single():
    assert natural_list(["one"]) == "one"


def test_natural_list_two_items():
    assert natural_list(["one", "two"]) == "one and two"


def test_natural_list_three_items():
    assert natural_list(["one", "two", "three"]) == "one, two and three"


def test_natural_list_uses_str_on_items():
    assert natural_list([1, Path("x"), None]) == "1, x and None"


def test_ordinal_basic_cases():
    assert ordinal(1) == "1st"
    assert ordinal(2) == "2nd"
    assert ordinal(3) == "3rd"
    assert ordinal(4) == "4th"


def test_ordinal_teens_use_th():
    assert ordinal(11) == "11th"
    assert ordinal(12) == "12th"
    assert ordinal(13) == "13th"


def test_ordinal_non_numeric_passthrough():
    assert ordinal("something else") == "something else"


def test_ordinal_handles_negative_numbers():
    assert ordinal(-1) == "-1st"


def test_ordinal_handles_infinite_and_nan():
    assert ordinal(float("inf")) == "+Inf"
    assert ordinal(float("-inf")) == "-Inf"
    assert ordinal(float("nan")) == "NaN"


def test_apnumber_single_digits_are_words():
    assert apnumber(0) == "zero"
    assert apnumber(5) == "five"
    assert apnumber("7") == "seven"


def test_apnumber_outside_range_returns_string_number():
    assert apnumber(10) == "10"
    assert apnumber(-1) == "-1"


def test_apnumber_non_numeric_passthrough():
    assert apnumber("foo") == "foo"


def test_intcomma_basic_integers():
    assert intcomma(1000) == "1,000"
    assert intcomma(1000000) == "1,000,000"


def test_intcomma_float_and_rounding_digits():
    assert intcomma(1234.5454545, 2) == "1,234.55"
    assert intcomma(14308.40, 1) == "14,308.4"


def test_intcomma_string_input_with_decimal_and_thousands():
    assert intcomma("14308.40", 1) == "14,308.4"


def test_intcomma_none_passthrough_as_string():
    assert intcomma(None) == "None"


def test_intcomma_handles_infinite_and_nan():
    assert intcomma(float("inf")) == "+Inf"
    assert intcomma(float("-inf")) == "-Inf"
    assert intcomma(float("nan")) == "NaN"


def test_intcomma_with_locale_separators():
    activate("de_DE", path=".")
    try:
        assert thousands_separator() == "."
        assert decimal_separator() == ","
        assert intcomma("1.234,5") == "1.234,5"
    finally:
        deactivate()


def test_intword_small_number_returns_original():
    assert intword("100") == "100"


def test_intword_thousand_and_million():
    assert intword("12400") == "12.4 thousand"
    assert intword("1000000") == "1.0 million"


def test_intword_large_numbers_and_negative():
    assert intword(1_200_000_000) == "1.2 billion"
    assert intword(-12_400_000) == "-12.4 million"


def test_intword_custom_format():
    assert intword("1234000", "%0.3f") == "1.234 million"


def test_intword_handles_infinite_and_nan():
    assert intword(float("inf")) == "+Inf"
    assert intword(float("nan")) == "NaN"


def test_fractional_simple_cases():
    assert fractional(0.3) == "3/10"
    assert fractional(1.3) == "1 3/10"
    assert fractional(1) == "1"


def test_fractional_negative_and_repeating():
    assert fractional(-1.3) == "-1 3/10"
    assert fractional(float(1 / 3)) == "1/3"


def test_fractional_non_numeric_passthrough():
    assert fractional("ten") == "ten"
    assert fractional(None) == "None"


def test_scientific_simple_cases():
    assert scientific(500) == "5.00 x 10²"
    assert scientific(-1000) == "-1.00 x 10³"
    assert scientific(1000, 1) == "1.0 x 10³"


def test_scientific_string_and_non_numeric_passthrough():
    assert scientific("99") == "9.90 x 10¹"
    assert scientific("foo") == "foo"


def test_scientific_handles_infinite_and_nan():
    assert scientific(float("inf")) == "+Inf"
    assert scientific(float("-inf")) == "-Inf"
    assert scientific(float("nan")) == "NaN"


def test_clamp_no_clamping_returns_formatted_value():
    assert clamp(123.456) == "123.456"


def test_clamp_floor_and_ceil():
    assert clamp(0.0001, floor=0.01) == "<0.01"
    assert clamp(0.999, format="{:.0%}", ceil=0.99) == ">99%"


def test_clamp_callable_format_and_custom_tokens():
    assert clamp(1, format=intword, floor=1e6, floor_token="under ") == "under 1.0 million"


def test_clamp_none_and_nonfinite():
    assert clamp(None) is None
    assert clamp(float("inf")) == "+Inf"


def test_clamp_invalid_format_raises():
    with pytest.raises(ValueError):
        clamp(1, format=object())


def test_metric_decimal_prefix():
    assert metric(1500, "V") == "1.50 kV"


def test_metric_large_and_micro_prefixes():
    assert metric(2e8, "W") == "200 MW"
    assert metric(220e-6, "F") == "220 μF"


def test_metric_scientific_fallback():
    assert metric(1e40) == "1.00 x 10⁴⁰"


def test_metric_zero_and_negative():
    assert metric(0, "m") == "0.00 m"
    assert metric(-1500, "V") == "-1.50 kV"


def test_naturalsize_bytes_and_units():
    assert naturalsize(1) == "1 Byte"
    assert naturalsize(2) == "2 Bytes"
    assert naturalsize(3000000) == "3.0 MB"


def test_naturalsize_binary_and_gnu():
    assert naturalsize(3000, True) == "2.9 KiB"
    assert naturalsize(300, False, True) == "300B"
    assert naturalsize(3000, False, True, "%.3f") == "2.930K"


def test_naturalsize_negative_and_large_suffix():
    assert naturalsize(-4096, True) == "-4.0 KiB"
    assert naturalsize(10**28) == "10.0 RB"


def test_naturalsize_rounding_bumps_suffix():
    assert naturalsize(999_999) == "1.0 MB"


def test_naturaldelta_basic_seconds_and_minutes():
    assert naturaldelta(1) == "a second"
    assert naturaldelta(59) == "59 seconds"
    assert naturaldelta(60) == "a minute"
    assert naturaldelta(119) == "2 minutes"


def test_naturaldelta_hours_days_and_months():
    assert naturaldelta(3600) == "an hour"
    assert naturaldelta(86400) == "a day"
    assert naturaldelta(31 * 86400) == "a month"


def test_naturaldelta_years_and_composite():
    assert naturaldelta(365 * 86400) == "a year"
    assert naturaldelta(400 * 86400) == "1 year, 1 month"


def test_naturaldelta_minimum_unit_microseconds_and_milliseconds():
    assert naturaldelta(dt.timedelta(microseconds=123), minimum_unit="microseconds") == "123 microseconds"
    assert naturaldelta(dt.timedelta(microseconds=123000), minimum_unit="microseconds") == "123 milliseconds"


def test_naturaldelta_invalid_minimum_unit_raises():
    with pytest.raises(ValueError):
        naturaldelta(1, minimum_unit="minutes")


def test_naturaldelta_non_numeric_passthrough():
    assert naturaldelta("foo") == "foo"


def test_naturaltime_past_and_future_with_numbers():
    assert naturaltime(1) == "1 second ago"
    assert naturaltime(1, future=True) == "1 second from now"


def test_naturaltime_datetime_past_and_future():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.datetime(2020, 1, 1, 11, 59, 0), when=now) == "a minute ago"
    assert naturaltime(dt.datetime(2020, 1, 1, 12, 1, 0), when=now) == "a minute from now"


def test_naturaltime_now_for_moment():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(0, when=now) == "now"


def test_naturaltime_timedelta_uses_its_own_tense():
    now = dt.datetime(2020, 1, 1, 12, 0, 0)
    assert naturaltime(dt.timedelta(minutes=2), when=now) == "2 minutes ago"


def test_naturaltime_timezone_aware_datetime():
    tz = dt.timezone(dt.timedelta(hours=2))
    now = dt.datetime(2020, 1, 1, 12, 0, 0, tzinfo=tz)
    later = dt.datetime(2020, 1, 1, 12, 2, 0, tzinfo=tz)
    assert naturaltime(later, when=now) == "2 minutes from now"


def test_naturalday_today_tomorrow_yesterday():
    with freeze_time("2020-01-02"):
        assert naturalday(dt.date(2020, 1, 2)) == "today"
        assert naturalday(dt.date(2020, 1, 3)) == "tomorrow"
        assert naturalday(dt.date(2020, 1, 1)) == "yesterday"


def test_naturalday_formats_other_dates():
    with freeze_time("2020-01-02"):
        assert naturalday(dt.date(2020, 1, 10)) == "Jan 10"


def test_naturalday_non_date_passthrough():
    assert naturalday("foo") == "foo"


def test_naturaldate_today_and_far_away():
    with freeze_time("2020-01-02"):
        assert naturaldate(dt.date(2020, 1, 2)) == "today"
        assert naturaldate(dt.date(2021, 1, 10)) == "Jan 10 2021"


def test_naturaldate_non_date_passthrough():
    assert naturaldate("foo") == "foo"


def test_precisedelta_simple_and_compound():
    delta = dt.timedelta(days=2, hours=1, minutes=33, seconds=0, microseconds=120000)
    assert precisedelta(delta) == "2 days, 1 hour and 33.12 seconds"


def test_precisedelta_custom_format():
    delta = dt.timedelta(days=2, hours=1, minutes=33, microseconds=123000)
    assert precisedelta(delta, format="%0.4f") == "2 days, 1 hour and 33.1230 seconds"


def test_precisedelta_microseconds_minimum_unit():
    delta = dt.timedelta(days=2, hours=1, minutes=33, microseconds=123000)
    assert precisedelta(delta, minimum_unit="microseconds") == "2 days, 1 hour, 33 seconds and 123 milliseconds"


def test_precisedelta_suppress_days():
    delta = dt.timedelta(days=2, hours=1, minutes=33, microseconds=120000)
    assert precisedelta(delta, suppress=["days"]) == "49 hours and 33.12 seconds"


def test_precisedelta_suppress_seconds_and_smaller():
    delta = dt.timedelta(seconds=90, microseconds=100)
    assert precisedelta(delta, suppress=["seconds", "milliseconds", "microseconds"]) == "1.50 minutes"


def test_precisedelta_minimum_unit_more_coarse_than_value():
    delta = dt.timedelta(seconds=1)
    assert precisedelta(delta, minimum_unit="minutes") == "0.02 minutes"


def test_precisedelta_tiny_value_with_coarse_minimum_unit():
    delta = dt.timedelta(seconds=0.1)
    assert precisedelta(delta, minimum_unit="minutes") == "0 minutes"


def test_precisedelta_none_passthrough():
    assert precisedelta(None) is None


def test_precisedelta_non_numeric_passthrough():
    assert precisedelta("foo") == "foo"


def test_rounding_by_fmt_int_and_float():
    assert _rounding_by_fmt("%d", 1.9) == 1
    assert _rounding_by_fmt("%.2f", 1.234) == 1.23


def test_suitable_minimum_unit_and_suppressed_units_helpers():
    assert _suitable_minimum_unit(Unit.HOURS, []).name == "HOURS"
    assert _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS]).name == "DAYS"
    assert [x.name for x in sorted(_suppress_lower_units(Unit.SECONDS, [Unit.DAYS]))] == [
        "MICROSECONDS",
        "MILLISECONDS",
        "DAYS",
    ]


def test_format_not_finite_helper():
    assert _format_not_finite(float("nan")) == "NaN"
    assert _format_not_finite(float("-inf")) == "-Inf"
    assert _format_not_finite(float("inf")) == "+Inf"


def test_activate_deactivate_and_separators_default_locale():
    deactivate()
    assert decimal_separator() == "."
    assert thousands_separator() == ","
    assert activate(None) is not None
    assert decimal_separator() == "."
    assert thousands_separator() == ","


def test_activate_english_locale_disables_translation():
    assert activate("en_GB") is not None
    assert decimal_separator() == "."
    assert thousands_separator() == ","


def test_activate_unknown_locale_raises_file_not_found_when_no_path():
    with pytest.raises(FileNotFoundError):
        activate("xx_YY", path=None)


def test_gettext_noop_helpers_are_identity():
    assert i18n._gettext_noop("abc") == "abc"
    assert i18n._ngettext_noop("a", "b") == ("a", "b")


def test_gettext_helpers_use_active_translation():
    class DummyTranslation(gettext.NullTranslations):
        def gettext(self, message):
            return f"T:{message}"

        def ngettext(self, singular, plural, n):
            return f"T:{singular if n == 1 else plural}"

        def pgettext(self, context, message):
            return f"T:{context}:{message}"

    i18n._TRANSLATIONS["zz_ZZ"] = DummyTranslation()
    try:
        activate("zz_ZZ", path=".")
        assert i18n._gettext("hello") == "T:hello"
        assert i18n._ngettext("one", "many", 2) == "T:many"
        assert i18n._pgettext("ctx", "hello") == "T:ctx:hello"
    finally:
        deactivate()
        i18n._TRANSLATIONS.pop("zz_ZZ", None)
