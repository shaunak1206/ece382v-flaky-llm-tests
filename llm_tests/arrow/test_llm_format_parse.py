"""Unit tests for Arrow.format, tokens, parser behaviour and isoformat/timestamp output."""
from datetime import datetime, timezone

import pytest

import arrow
from arrow import Arrow
from arrow.parser import ParserError, ParserMatchError


@pytest.fixture
def sample():
    return Arrow(2013, 5, 5, 15, 7, 9, 123456, tzinfo="US/Pacific")


@pytest.mark.parametrize(
    "fmt, expected",
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
        ("HH", "15"),
        ("H", "15"),
        ("hh", "03"),
        ("h", "3"),
        ("mm", "07"),
        ("m", "7"),
        ("ss", "09"),
        ("s", "9"),
        ("S", "1"),
        ("SS", "12"),
        ("SSS", "123"),
        ("SSSSSS", "123456"),
        ("a", "pm"),
        ("A", "PM"),
        ("ZZ", "-07:00"),
        ("Z", "-0700"),
        ("W", "2013-W18-7"),
    ],
)
def test_format_tokens(sample, fmt, expected):
    assert sample.format(fmt) == expected


def test_format_default():
    assert Arrow(2013, 5, 5, 3, 4, 5).format() == "2013-05-05 03:04:05+00:00"


def test_format_escaped_text_in_brackets(sample):
    assert sample.format("[Today is] dddd") == "Today is Sunday"


def test_format_full_sentence(sample):
    assert sample.format("dddd, MMMM Do YYYY [at] h:mm a") == (
        "Sunday, May 5th 2013 at 3:07 pm"
    )


def test_format_timestamp_tokens():
    a = Arrow(2013, 5, 5, 12)
    assert float(a.format("X")) == 1367755200.0
    assert a.format("x") == "1367755200000000"


def test_format_in_french_and_german_locales():
    a = Arrow(2013, 5, 5)
    assert a.format("dddd D MMMM YYYY", locale="fr") == "dimanche 5 mai 2013"
    assert a.format("MMMM", locale="de") == "Mai"
    assert a.format("dddd", locale="es") == "domingo"


def test_isoformat_variants():
    a = Arrow(2013, 5, 5, 12, 30, 45, 500000, tzinfo="Asia/Tokyo")
    assert a.isoformat() == "2013-05-05T12:30:45.500000+09:00"
    assert a.isoformat(sep=" ", timespec="seconds") == "2013-05-05 12:30:45+09:00"


def test_timestamp_values():
    a = Arrow(2013, 5, 5, 12, 0, 0, 500000)
    assert a.timestamp() == 1367755200.5
    assert a.int_timestamp == 1367755200
    assert a.float_timestamp == 1367755200.5


@pytest.mark.parametrize(
    "text, fmt, expected",
    [
        ("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss", (2013, 5, 5, 12, 30, 45)),
        ("05/06/2013", "MM/DD/YYYY", (2013, 5, 6, 0, 0, 0)),
        ("May 5, 2013", "MMMM D, YYYY", (2013, 5, 5, 0, 0, 0)),
        ("Sun, 05 May 13", "ddd, DD MMM YY", (2013, 5, 5, 0, 0, 0)),
        ("2013-125", "YYYY-DDDD", (2013, 5, 5, 0, 0, 0)),
        ("3:07 pm 2013-05-05", "h:mm a YYYY-MM-DD", (2013, 5, 5, 15, 7, 0)),
        ("5th of May 2013", "Do [of] MMMM YYYY", (2013, 5, 5, 0, 0, 0)),
    ],
)
def test_parse_with_format_tokens(text, fmt, expected):
    parsed = arrow.get(text, fmt)
    assert (
        parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second
    ) == expected


def test_parse_timestamp_token():
    assert arrow.get("1367755200", "X") == Arrow(2013, 5, 5, 12)


def test_parse_fractional_seconds():
    parsed = arrow.get("2013-05-05 12:30:45.123456", "YYYY-MM-DD HH:mm:ss.S")
    assert parsed.microsecond == 123456


def test_parse_french_month_names():
    parsed = arrow.get("5 mai 2013", "D MMMM YYYY", locale="fr")
    assert (parsed.year, parsed.month, parsed.day) == (2013, 5, 5)


def test_parse_format_mismatch_raises():
    with pytest.raises(ParserMatchError):
        arrow.get("2013-05-05", "HH:mm")


def test_parse_out_of_range_values_raise():
    with pytest.raises(ValueError):
        arrow.get("2013-13-45")


def test_format_then_parse_roundtrip():
    original = Arrow(2021, 11, 7, 8, 9, 10, tzinfo="Asia/Kolkata")
    fmt = "YYYY-MM-DD HH:mm:ss ZZ"
    assert arrow.get(original.format(fmt), fmt) == original
