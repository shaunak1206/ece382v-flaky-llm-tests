"""Tests for arrow.parser and arrow.formatter (via the public Arrow API where possible)."""

from datetime import datetime, timedelta, timezone

import pytest

from arrow import Arrow
from arrow import formatter as fmt_module
from arrow.formatter import DateTimeFormatter
from arrow.parser import DateTimeParser, ParserError, ParserMatchError, TzinfoParser

UTC = timezone.utc


@pytest.fixture
def parser():
    return DateTimeParser()


# ---------------------------------------------------------------------------
# parser: parse_iso
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-05-05", datetime(2013, 5, 5)),
        ("2013", datetime(2013, 1, 1)),
        ("2013-05", datetime(2013, 5, 1)),
        ("2013/05/06", datetime(2013, 5, 6)),
        ("2013.05.07", datetime(2013, 5, 7)),
        ("20130505", datetime(2013, 5, 5)),
    ],
)
def test_parse_iso_date_formats(parser, text, expected):
    assert parser.parse_iso(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-05-05T12:30:45", datetime(2013, 5, 5, 12, 30, 45)),
        ("2013-05-05 12:30", datetime(2013, 5, 5, 12, 30)),
        ("2013-05-05T07", datetime(2013, 5, 5, 7)),
        ("20130505T123045", datetime(2013, 5, 5, 12, 30, 45)),
    ],
)
def test_parse_iso_datetime_formats(parser, text, expected):
    assert parser.parse_iso(text) == expected


@pytest.mark.parametrize(
    "text, offset",
    [
        ("2013-05-05T12:30:45+02:00", timedelta(hours=2)),
        ("2013-05-05T12:30:45-0730", -timedelta(hours=7, minutes=30)),
        ("2013-05-05T12:30:45Z", timedelta(0)),
    ],
)
def test_parse_iso_timezone_offsets(parser, text, offset):
    result = parser.parse_iso(text)
    assert result.utcoffset() == offset
    assert (result.hour, result.minute, result.second) == (12, 30, 45)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2013-036", datetime(2013, 2, 5)),
        ("2013036", datetime(2013, 2, 5)),
        ("2013-W05-5", datetime(2013, 2, 1)),
        ("2013W055", datetime(2013, 2, 1)),
        ("2013-W05", datetime(2013, 1, 28)),
    ],
)
def test_parse_iso_ordinal_and_week_dates(parser, text, expected):
    assert parser.parse_iso(text) == expected


@pytest.mark.parametrize(
    "text, microsecond",
    [
        ("2013-05-05T12:30:45.1234567", 123457),
        ("2013-05-05T12:30:45.1234565", 123456),
        ("2013-05-05T12:30:45.1234575", 123458),
        ("2013-05-05T12:30:45,5", 500000),
    ],
)
def test_parse_iso_subsecond_rounding(parser, text, microsecond):
    assert parser.parse_iso(text).microsecond == microsecond


def test_parse_iso_subsecond_rounding_carries_into_next_second(parser):
    result = parser.parse_iso("2013-05-05T12:30:45.9999996")
    assert result == datetime(2013, 5, 5, 12, 30, 46)


def test_parse_iso_midnight_at_end_of_day(parser):
    assert parser.parse_iso("2013-05-05T24:00:00") == datetime(2013, 5, 6)


@pytest.mark.parametrize(
    "text",
    [
        "2013-05-05T24:01:00",
        "2013-05-05  12:30:45",
        "2013-05-05T1:2",
        "not-a-date",
        "2013-02-30",
    ],
)
def test_parse_iso_invalid_input_raises(parser, text):
    with pytest.raises(ValueError):
        parser.parse_iso(text)


def test_parse_iso_normalize_whitespace_collapses_spaces(parser):
    result = parser.parse_iso("  2013-05-05   12:30:45 ", normalize_whitespace=True)
    assert result == datetime(2013, 5, 5, 12, 30, 45)


# ---------------------------------------------------------------------------
# parser: parse with explicit formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, fmt, expected",
    [
        ("2021-10-12 14:30:00", "YYYY-MM-DD HH:mm:ss", datetime(2021, 10, 12, 14, 30)),
        ("May 5th, 2013", "MMMM Do, YYYY", datetime(2013, 5, 5)),
        ("5 mAr 2013", "D MMM YYYY", datetime(2013, 3, 5)),
        ("2013-05-05 at 12:30", "YYYY-MM-DD [at] HH:mm", datetime(2013, 5, 5, 12, 30)),
        ("The date is 2013-05-05, thanks", "YYYY-MM-DD", datetime(2013, 5, 5)),
        ("2013 036", "YYYY DDDD", datetime(2013, 2, 5)),
        ("2013 36", "YYYY DDD", datetime(2013, 2, 5)),
    ],
)
def test_parse_with_format_tokens(parser, text, fmt, expected):
    assert parser.parse(text, fmt) == expected


@pytest.mark.parametrize(
    "text, year",
    [("99-01-01", 1999), ("69-01-01", 1969), ("68-01-01", 2068), ("05-01-01", 2005)],
)
def test_parse_two_digit_year_pivot(parser, text, year):
    assert parser.parse(text, "YY-MM-DD").year == year


def test_parse_twelve_hour_clock(parser):
    assert parser.parse("2013-05-05 03:30 pm", "YYYY-MM-DD hh:mm a").hour == 15
    assert parser.parse("2013-05-05 12:30 am", "YYYY-MM-DD hh:mm a").hour == 0
    assert parser.parse("2013-05-05 12:30 PM", "YYYY-MM-DD hh:mm A").hour == 12


def test_parse_am_with_hour_above_twelve_raises(parser):
    with pytest.raises(ParserMatchError):
        parser.parse("2013-05-05 13:00 am", "YYYY-MM-DD HH:mm a")


@pytest.mark.parametrize("text, fmt", [("1368071882", "X"), ("1368071882000", "x")])
def test_parse_timestamp_tokens(parser, text, fmt):
    assert parser.parse(text, fmt) == datetime(2013, 5, 9, 3, 58, 2, tzinfo=UTC)


def test_parse_timezone_name_token(parser):
    result = parser.parse("2013-05-05 12:30 Europe/Berlin", "YYYY-MM-DD HH:mm ZZZ")
    assert result.utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize(
    "text, fmt",
    [("036", "DDDD"), ("2013 02 036", "YYYY MM DDDD"), ("2013 400", "YYYY DDDD")],
)
def test_parse_invalid_day_of_year_usage_raises(parser, text, fmt):
    with pytest.raises(ParserError):
        parser.parse(text, fmt)


def test_parse_weekday_names(parser):
    # 1970-01-01 was a Thursday, so the first Monday on or after it is 1970-01-05.
    assert parser.parse("Monday", "dddd") == datetime(1970, 1, 5)
    # an explicit day wins over the weekday name
    assert parser.parse("Mon 2013-05-08", "ddd YYYY-MM-DD") == datetime(2013, 5, 8)


@pytest.mark.parametrize(
    "text, fmt",
    [("blah2013-05-05", "YYYY-MM-DD"), ("2013-05", "YYYY-MM-DD"), ("hello", "YYYY")],
)
def test_parse_non_matching_input_raises_match_error(parser, text, fmt):
    with pytest.raises(ParserMatchError):
        parser.parse(text, fmt)


def test_parse_with_format_list(parser):
    assert parser.parse("05/06/2013", ["YYYY-MM-DD", "MM/DD/YYYY"]) == datetime(2013, 5, 6)
    with pytest.raises(ParserError):
        parser.parse("hello", ["YYYY-MM-DD", "MM/DD/YYYY"])


def test_parse_normalize_whitespace_and_cache_size():
    cached = DateTimeParser(cache_size=4)
    result = cached.parse(
        "2013-05-05   12:30", "YYYY-MM-DD HH:mm", normalize_whitespace=True
    )
    assert result == datetime(2013, 5, 5, 12, 30)
    assert issubclass(ParserError, ValueError)
    assert issubclass(ParserMatchError, ParserError)


# ---------------------------------------------------------------------------
# TzinfoParser
# ---------------------------------------------------------------------------


def test_tzinfo_parser_utc_aliases_and_offsets():
    for alias in ("utc", "UTC", "Z"):
        assert TzinfoParser.parse(alias) == timezone.utc
    assert TzinfoParser.parse("+05:30").utcoffset(None) == timedelta(hours=5, minutes=30)
    assert TzinfoParser.parse("-0800").utcoffset(None) == timedelta(hours=-8)
    assert TzinfoParser.parse("+09").utcoffset(None) == timedelta(hours=9)


def test_tzinfo_parser_iana_names_and_invalid_input():
    tz = TzinfoParser.parse("Europe/Berlin")
    assert Arrow(2013, 1, 1, tzinfo=tz).utcoffset() == timedelta(hours=1)
    assert Arrow(2013, 7, 1, tzinfo=tz).utcoffset() == timedelta(hours=2)
    with pytest.raises(ParserError):
        TzinfoParser.parse("Not/A_Real_Zone")


# ---------------------------------------------------------------------------
# formatter
# ---------------------------------------------------------------------------

# 2021-02-03 is a Wednesday, day 34 of the year.
SAMPLE = Arrow(2021, 2, 3, 4, 5, 6, 123456)


def test_format_default_and_dunder_format():
    assert Arrow(2021, 2, 3, 4, 5, 6).format() == "2021-02-03 04:05:06+00:00"
    assert f"{SAMPLE:YYYY-MM-DD}" == "2021-02-03"
    assert f"{SAMPLE}" == str(SAMPLE)


@pytest.mark.parametrize(
    "token, expected",
    [
        ("YYYY", "2021"),
        ("YY", "21"),
        ("MMMM", "February"),
        ("MMM", "Feb"),
        ("MM", "02"),
        ("M", "2"),
        ("DDDD", "034"),
        ("DDD", "34"),
        ("DD", "03"),
        ("D", "3"),
        ("Do", "3rd"),
        ("dddd", "Wednesday"),
        ("ddd", "Wed"),
        ("d", "3"),
    ],
)
def test_format_date_tokens(token, expected):
    assert SAMPLE.format(token) == expected


@pytest.mark.parametrize(
    "day, expected",
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (22, "22nd")],
)
def test_format_ordinal_suffixes(day, expected):
    assert Arrow(2021, 1, day).format("Do") == expected


def test_format_hour_minute_second_tokens():
    late = Arrow(2021, 2, 3, 9, 8, 7)
    assert late.format("HH:mm:ss") == "09:08:07"
    assert late.format("H:m:s") == "9:8:7"
    assert Arrow(2021, 2, 3, 15).format("h") == "3"
    assert late.format("hh") == "09"
    assert Arrow(2021, 2, 3, 0).format("h") == "12"
    assert Arrow(2021, 2, 3, 12).format("hh") == "12"


def test_format_meridian_tokens():
    assert Arrow(2021, 2, 3, 15).format("a A") == "pm PM"
    assert Arrow(2021, 2, 3, 9).format("a A") == "am AM"


def test_format_subsecond_tokens():
    expected = {
        "S": "1",
        "SS": "12",
        "SSS": "123",
        "SSSS": "1234",
        "SSSSS": "12345",
        "SSSSSS": "123456",
    }
    for token, text in expected.items():
        assert SAMPLE.format(token) == text


def test_format_timezone_timestamp_and_week_tokens():
    ist = Arrow(2021, 2, 3, tzinfo="+05:30")
    assert ist.format("ZZ") == "+05:30"
    assert ist.format("Z") == "+0530"
    west = Arrow(2021, 2, 3, tzinfo="-08:00")
    assert west.format("ZZ") == "-08:00"
    assert west.format("Z") == "-0800"
    assert SAMPLE.format("ZZ Z ZZZ") == "+00:00 +0000 UTC"
    moment = Arrow(2013, 5, 8, 5, 54, 34)
    assert moment.format("X") == str(moment.timestamp())
    assert moment.format("x") == "1367992474000000"
    assert Arrow(2013, 5, 5).format("W") == "2013-W18-7"


def test_format_escaped_text_and_plain_punctuation():
    assert SAMPLE.format("YYYY [YYYY MM] DD") == "2021 YYYY MM 03"
    assert SAMPLE.format("YYYY/MM/DD @ HH.mm") == "2021/02/03 @ 04.05"


def test_format_locale_support():
    assert SAMPLE.format("dddd D MMMM YYYY", locale="fr") == "mercredi 3 février 2021"
    with pytest.raises(ValueError):
        SAMPLE.format("YYYY", locale="xx-nowhere")


def test_formatter_constants_render_expected_strings():
    moment = Arrow(2021, 2, 3, 4, 5, 6)
    assert moment.format(fmt_module.FORMAT_RFC2822) == "Wed, 03 Feb 2021 04:05:06 +0000"
    assert moment.format(fmt_module.FORMAT_RFC3339_STRICT) == "2021-02-03T04:05:06+00:00"
    assert moment.format(fmt_module.FORMAT_ATOM) == "2021-02-03 04:05:06+00:00"
    assert moment.format(fmt_module.FORMAT_COOKIE) == "Wednesday, 03-Feb-2021 04:05:06 UTC"


def test_formatter_class_formats_plain_datetime():
    formatter = DateTimeFormatter()
    dt = datetime(2020, 12, 31, 23, 59, 58, tzinfo=timezone(timedelta(hours=1)))
    assert formatter.format(dt, "YYYY-MM-DD HH:mm:ssZZ") == "2020-12-31 23:59:58+01:00"
