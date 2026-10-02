"""Tests for formatting, parsing, isoformat, timestamps and calendar helpers."""

import calendar
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import arrow
from arrow import Arrow
from arrow.parser import DateTimeParser, ParserError, ParserMatchError, TzinfoParser


@pytest.fixture
def base():
    # Sunday, ISO week 18, day-of-year 125
    return Arrow(2013, 5, 5, 12, 30, 45, 123456)


UTC = timezone.utc


# ---------------------------------------------------------------- format


def test_format_default_pattern(base):
    assert base.format() == "2013-05-05 12:30:45+00:00"
    assert Arrow(2013, 5, 5, 12, tzinfo="US/Pacific").format() == "2013-05-05 12:00:00-07:00"


@pytest.mark.parametrize(
    "token, expected",
    [
        ("YYYY", "2013"),
        ("YY", "13"),
        ("MMMM", "May"),
        ("MMM", "May"),
        ("MM", "05"),
        ("M", "5"),
        ("DDDD", "125"),
        ("DDD", "125"),
        ("DD", "05"),
        ("D", "5"),
        ("Do", "5th"),
        ("dddd", "Sunday"),
        ("ddd", "Sun"),
        ("d", "7"),
        ("HH", "12"),
        ("H", "12"),
        ("hh", "12"),
        ("h", "12"),
        ("mm", "30"),
        ("m", "30"),
        ("ss", "45"),
        ("s", "45"),
        ("S", "1"),
        ("SS", "12"),
        ("SSS", "123"),
        ("SSSS", "1234"),
        ("SSSSS", "12345"),
        ("SSSSSS", "123456"),
        ("a", "pm"),
        ("A", "PM"),
        ("ZZ", "+00:00"),
        ("Z", "+0000"),
        ("ZZZ", "UTC"),
        ("W", "2013-W18-7"),
    ],
)
def test_format_tokens(base, token, expected):
    assert base.format(token) == expected


def test_format_combined_patterns_timestamps_and_timezones(base):
    assert base.format("YYYYMMDD") == "20130505"
    assert base.format("HHmmss") == "123045"
    assert base.format("dddd, MMMM D, YYYY") == "Sunday, May 5, 2013"
    assert base.format("YYYY年MM月DD日") == "2013年05月05日"
    whole = Arrow(2013, 5, 5, 12, 30, 45)
    assert float(whole.format("X")) == 1367757045.0
    assert whole.format("x") == "1367757045000000"
    assert Arrow(2013, 1, 5).format("DDDD") == "005"

    summer = Arrow(2013, 7, 1, 12, tzinfo="US/Pacific")
    winter = Arrow(2013, 1, 1, 12, tzinfo="US/Pacific")
    assert summer.format("ZZ Z ZZZ") == "-07:00 -0700 PDT"
    assert winter.format("ZZ Z ZZZ") == "-08:00 -0800 PST"
    assert Arrow(2013, 7, 1, 12, tzinfo="Asia/Kolkata").format("ZZ Z") == "+05:30 +0530"
    assert Arrow(2013, 7, 1, 12, tzinfo="-02:30").format("ZZ Z") == "-02:30 -0230"


def test_format_twelve_hour_clock_and_escapes(base):
    assert Arrow(2013, 12, 25, 0, 5, 9).format("h:mm:ss A") == "12:05:09 AM"
    assert Arrow(2013, 12, 25, 1, 5, 9).format("hh:mm a") == "01:05 am"
    assert Arrow(2013, 12, 25, 12, 0, 0).format("h:mm A") == "12:00 PM"
    assert Arrow(2013, 12, 25, 13, 7, 0).format("h:mm A") == "1:07 PM"
    assert Arrow(2013, 12, 25, 23, 59, 0).format("hh:mm A") == "11:59 PM"

    assert base.format("[Today is] dddd") == "Today is Sunday"
    assert base.format("YYYY [at] HH:mm") == "2013 at 12:30"
    assert base.format("[YYYY]") == "YYYY"
    assert base.format("[W] W") == "W 2013-W18-7"


def test_format_in_other_locales(base):
    assert base.format("dddd D MMMM YYYY", locale="fr") == "dimanche 5 mai 2013"
    assert base.format("dddd, Do MMMM YYYY", locale="de") == "Sonntag, 5. Mai 2013"
    assert base.format("dddd, D [de] MMMM [de] YYYY", locale="es") == (
        "domingo, 5 de mayo de 2013"
    )
    assert base.format("MMM ddd", locale="fr") == "mai dim"
    assert base.format("dddd", locale="en_us") == "Sunday"
    with pytest.raises(ValueError):
        base.format("YYYY", locale="xx-yy")


def test_str_repr_format_spec_and_named_formats(base):
    assert str(base) == "2013-05-05T12:30:45.123456+00:00"
    assert repr(base) == "<Arrow [2013-05-05T12:30:45.123456+00:00]>"
    assert f"{base}" == "2013-05-05T12:30:45.123456+00:00"
    assert f"{base:YYYY-MM-DD}" == "2013-05-05"
    assert base.format(arrow.FORMAT_ATOM) == "2013-05-05 12:30:45+00:00"
    assert base.format(arrow.FORMAT_RFC2822) == "Sun, 05 May 2013 12:30:45 +0000"
    assert base.format(arrow.FORMAT_RFC3339_STRICT) == "2013-05-05T12:30:45+00:00"
    assert base.format(arrow.FORMAT_COOKIE) == "Sunday, 05-May-2013 12:30:45 UTC"
    assert base.format(arrow.FORMAT_RFC850) == "Sunday, 05-May-13 12:30:45 UTC"


# ---------------------------------------------------------------- isoformat / timestamps


def test_isoformat_variants(base):
    assert base.isoformat() == "2013-05-05T12:30:45.123456+00:00"
    assert base.isoformat(sep=" ") == "2013-05-05 12:30:45.123456+00:00"
    assert base.isoformat(timespec="seconds") == "2013-05-05T12:30:45+00:00"
    assert base.isoformat(timespec="milliseconds") == "2013-05-05T12:30:45.123+00:00"
    assert base.for_json() == base.isoformat()
    assert Arrow(2013, 5, 5).isoformat() == "2013-05-05T00:00:00+00:00"
    assert base.to("US/Pacific").isoformat() == "2013-05-05T05:30:45.123456-07:00"
    assert arrow.get(base.isoformat()) == base


def test_timestamp_properties(base):
    assert base.timestamp() == pytest.approx(1367757045.123456)
    assert base.int_timestamp == 1367757045
    assert base.float_timestamp == base.timestamp()
    assert base.int_timestamp == calendar.timegm(base.utctimetuple())
    assert Arrow(1970, 1, 1).timestamp() == 0.0
    assert Arrow(1969, 12, 31, 23, 59, 59).int_timestamp == -1
    assert base.to("Asia/Tokyo").timestamp() == base.timestamp()
    assert abs((arrow.get(base.timestamp()) - base).total_seconds()) < 1e-3


# ---------------------------------------------------------------- parsing with formats


@pytest.mark.parametrize(
    "text, fmt, expected",
    [
        ("2013-05-05", "YYYY-MM-DD", datetime(2013, 5, 5)),
        ("05/06/13", "DD/MM/YY", datetime(2013, 6, 5)),
        ("May 5, 2013", "MMMM D, YYYY", datetime(2013, 5, 5)),
        ("May 5 2013", "MMM D YYYY", datetime(2013, 5, 5)),
        ("5th May 2013", "Do MMMM YYYY", datetime(2013, 5, 5)),
        ("Sunday, May 5 2013", "dddd, MMMM D YYYY", datetime(2013, 5, 5)),
        ("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss", datetime(2013, 5, 5, 12, 30, 45)),
        ("2013-05-05 1:30 PM", "YYYY-MM-DD h:mm A", datetime(2013, 5, 5, 13, 30)),
        ("2013-05-05 12:30 am", "YYYY-MM-DD hh:mm a", datetime(2013, 5, 5, 0, 30)),
        ("2013-05-05 12:30 pm", "YYYY-MM-DD hh:mm a", datetime(2013, 5, 5, 12, 30)),
        ("2013-125", "YYYY-DDDD", datetime(2013, 5, 5)),
        ("13-5-5 7:8:9", "YY-M-D H:m:s", datetime(2013, 5, 5, 7, 8, 9)),
        ("2013-W18-7", "W", datetime(2013, 5, 5)),
        ("1367757045", "X", datetime(2013, 5, 5, 12, 30, 45)),
        ("1367757045000", "x", datetime(2013, 5, 5, 12, 30, 45)),
        (
            "2013-05-05 12:30:45.123456",
            "YYYY-MM-DD HH:mm:ss.S",
            datetime(2013, 5, 5, 12, 30, 45, 123456),
        ),
        (
            "2013-05-05 12:30:45.123",
            "YYYY-MM-DD HH:mm:ss.SSS",
            datetime(2013, 5, 5, 12, 30, 45, 123000),
        ),
        ("2013-05-05T12:30", "YYYY-MM-DD[T]HH:mm", datetime(2013, 5, 5, 12, 30)),
        ("Date: 2013-05-05", "[Date:] YYYY-MM-DD", datetime(2013, 5, 5)),
    ],
)
def test_parse_with_format_tokens(text, fmt, expected):
    assert arrow.get(text, fmt).datetime == expected.replace(tzinfo=UTC)


def test_parse_two_digit_year_pivot_and_subsecond_rounding():
    assert arrow.get("69", "YY").year == 1969
    assert arrow.get("68", "YY").year == 2068
    fmt = "YYYY-MM-DD HH:mm:ss.S"
    assert arrow.get("2013-05-05 12:30:45.12345678", fmt).microsecond == 123457
    assert arrow.get("2013-05-05 12:30:45.5", fmt).microsecond == 500000
    assert arrow.get("2013-05-05 12:30:45.9999999", fmt).isoformat() == (
        "2013-05-05T12:30:46+00:00"
    )
    assert arrow.get("2013-05-05T24:00:00").isoformat() == "2013-05-06T00:00:00+00:00"


def test_parse_timezone_tokens():
    assert arrow.get("2013-05-05 12:30 +02:00", "YYYY-MM-DD HH:mm ZZ").utcoffset() == (
        timedelta(hours=2)
    )
    assert arrow.get("2013-05-05 12:30 -0530", "YYYY-MM-DD HH:mm Z").utcoffset() == (
        timedelta(hours=-5.5)
    )
    tokyo = arrow.get("2013-05-05 12:30 Asia/Tokyo", "YYYY-MM-DD HH:mm ZZZ")
    assert tokyo.isoformat() == "2013-05-05T12:30:00+09:00"
    assert arrow.get("2013-05-05 12:30 Z", "YYYY-MM-DD HH:mm Z").utcoffset() == timedelta(0)


def test_parse_finds_dates_inside_sentences():
    assert arrow.get("Meeting on 2013-05-05 at noon", "YYYY-MM-DD").date() == date(2013, 5, 5)
    assert arrow.get("Due 2013-05-05, thanks", "YYYY-MM-DD").date() == date(2013, 5, 5)
    with pytest.raises(ParserMatchError):
        arrow.get("blah2013-05-05", "YYYY-MM-DD")
    with pytest.raises(ParserMatchError):
        arrow.get("2013-05-05blah", "YYYY-MM-DD")


def test_parse_failures_raise_value_errors():
    with pytest.raises(ParserMatchError):
        arrow.get("2013-05-05", "HH:mm")
    with pytest.raises(ParserMatchError):
        arrow.get("abc", "YYYY")
    with pytest.raises(ValueError):
        arrow.get("2013-13-45", "YYYY-MM-DD")
    with pytest.raises(ValueError):
        arrow.get("2013-02-30", "YYYY-MM-DD")
    with pytest.raises(ValueError):
        arrow.get("25:00", "HH:mm")
    with pytest.raises(ParserError):
        arrow.get("24:30", "HH:mm")
    with pytest.raises(ParserError):
        arrow.get("125", "DDD")
    with pytest.raises(ParserError):
        arrow.get("2013-05-05T24:30:00")
    assert issubclass(ParserMatchError, ParserError)
    assert issubclass(ParserError, ValueError)


def test_parser_classes_directly():  # DateTimeParser and TzinfoParser
    parser = DateTimeParser()
    assert parser.parse_iso("2021-10-12T14:30:00") == datetime(2021, 10, 12, 14, 30)
    assert parser.parse("2021-10-12 14:30:00", "YYYY-MM-DD HH:mm:ss") == datetime(
        2021, 10, 12, 14, 30
    )
    assert parser.parse_iso("2013-05-05T12:30:45.123456+02:00").utcoffset() == timedelta(hours=2)
    assert DateTimeParser("fr").parse("5 mai 2013", "D MMMM YYYY") == datetime(2013, 5, 5)
    with pytest.raises(ParserError):
        parser.parse_iso("2013-05-05  12:30")
    with pytest.raises(ValueError):
        parser.parse_iso("2013-05-05T25:61")
    with pytest.raises(ValueError):
        DateTimeParser("xx-yy")

    assert TzinfoParser.parse("utc") == UTC
    assert TzinfoParser.parse("UTC") == UTC
    assert TzinfoParser.parse("Z") == UTC
    assert TzinfoParser.parse("+05:30") == timezone(timedelta(hours=5, minutes=30))
    assert TzinfoParser.parse("-0800") == timezone(timedelta(hours=-8))
    assert TzinfoParser.parse("+09") == timezone(timedelta(hours=9))
    assert TzinfoParser.parse("(UTC+02:00)") == timezone(timedelta(hours=2))
    assert TzinfoParser.parse("local").utcoffset(datetime.now()) == (
        datetime.now().astimezone().utcoffset()
    )
    pacific = TzinfoParser.parse("US/Pacific")
    assert isinstance(pacific, ZoneInfo)
    assert pacific.utcoffset(datetime(2013, 1, 1)) == timedelta(hours=-8)
    with pytest.raises(ParserError):
        TzinfoParser.parse("Not/AZone")


# ---------------------------------------------------------------- calendar helpers


def test_weekday_helpers(base):
    assert base.weekday() == 6
    assert base.isoweekday() == 7
    monday = Arrow(2013, 5, 6)
    days = [monday.shift(days=i) for i in range(7)]
    assert [d.weekday() for d in days] == [0, 1, 2, 3, 4, 5, 6]
    assert [d.isoweekday() for d in days] == [1, 2, 3, 4, 5, 6, 7]
    assert [d.format("ddd") for d in days] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@pytest.mark.parametrize(
    "when, expected",
    [
        (Arrow(2013, 5, 5), (2013, 18, 7)),
        (Arrow(2013, 5, 6), (2013, 19, 1)),
        (Arrow(2012, 12, 31), (2013, 1, 1)),
        (Arrow(2010, 1, 1), (2009, 53, 5)),
        (Arrow(2016, 1, 3), (2015, 53, 7)),
    ],
)
def test_isocalendar(when, expected):
    assert tuple(when.isocalendar()) == expected
    assert when.week == expected[1]


def test_datetime_style_accessors_and_quarters(base):
    quarters = {1: 1, 3: 1, 4: 2, 6: 2, 7: 3, 9: 3, 10: 4, 12: 4}
    for month, quarter in quarters.items():
        assert Arrow(2013, month, 1).quarter == quarter
    assert (base.year, base.month, base.day) == (2013, 5, 5)
    assert (base.hour, base.minute, base.second, base.microsecond) == (12, 30, 45, 123456)
    assert base.tzinfo == UTC
    assert base.fold == 0
    with pytest.raises(AttributeError):
        base.not_an_attribute

    assert base.date() == date(2013, 5, 5)
    assert base.time() == time(12, 30, 45, 123456)
    assert base.timetz() == time(12, 30, 45, 123456, tzinfo=UTC)
    assert base.naive == datetime(2013, 5, 5, 12, 30, 45, 123456)
    assert base.datetime == datetime(2013, 5, 5, 12, 30, 45, 123456, tzinfo=UTC)
    assert base.toordinal() == date(2013, 5, 5).toordinal()
    assert base.ctime() == "Sun May  5 12:30:45 2013"
    assert base.strftime("%Y/%m/%d %H") == "2013/05/05 12"
    assert base.timetuple().tm_yday == 125
    assert base.clone() == base
    assert base.clone() is not base

    pacific_summer = Arrow(2013, 5, 5, 12, tzinfo="US/Pacific")
    pacific_winter = Arrow(2013, 1, 5, 12, tzinfo="US/Pacific")
    assert pacific_summer.utcoffset() == timedelta(hours=-7)
    assert pacific_summer.dst() == timedelta(hours=1)
    assert pacific_winter.dst() == timedelta(0)
    assert pacific_summer.utctimetuple().tm_hour == 19
    assert base.astimezone(ZoneInfo("Asia/Tokyo")).hour == 21
