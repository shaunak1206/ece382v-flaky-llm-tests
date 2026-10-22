import datetime

import pytest

from arrow import Arrow
from arrow.formatter import DateTimeFormatter
from arrow.parser import (
    DateTimeParser,
    ParserError,
    ParserMatchError,
    TzinfoParser,
)

A = Arrow(2013, 5, 11, 20, 23, 58)  # Saturday


def test_timestamp_representations():
    assert A.timestamp() == 1368303838.0
    assert A.int_timestamp == 1368303838
    assert A.float_timestamp == 1368303838.0
    assert isinstance(A.int_timestamp, int)


def test_weekday_and_isocalendar():
    assert A.weekday() == 5
    assert A.isoweekday() == 6
    assert tuple(A.isocalendar()) == (2013, 19, 6)


def test_isoformat_strftime_ctime_for_json():
    assert A.isoformat() == "2013-05-11T20:23:58+00:00"
    assert A.isoformat(sep=" ") == "2013-05-11 20:23:58+00:00"
    assert A.strftime("%d-%m-%Y %H:%M:%S") == "11-05-2013 20:23:58"
    assert A.ctime() == "Sat May 11 20:23:58 2013"
    assert A.for_json() == A.isoformat()


def test_naive_date_time_and_timetz():
    pacific = Arrow(2013, 5, 11, 13, 23, 58, tzinfo="US/Pacific")
    assert pacific.naive == datetime.datetime(2013, 5, 11, 13, 23, 58)
    assert pacific.naive.tzinfo is None
    assert pacific.date() == datetime.date(2013, 5, 11)
    assert pacific.time() == datetime.time(13, 23, 58)
    assert pacific.timetz().tzinfo is not None
    assert pacific.datetime.tzinfo is not None


def test_ambiguous_imaginary_and_fold():
    ambiguous = Arrow(2013, 11, 3, 1, 30, tzinfo="US/Eastern")
    imaginary = Arrow(2013, 3, 10, 2, 30, tzinfo="US/Eastern")
    normal = Arrow(2013, 6, 1, 12, tzinfo="US/Eastern")
    assert ambiguous.ambiguous is True
    assert imaginary.imaginary is True
    assert normal.ambiguous is False
    assert normal.imaginary is False
    assert Arrow(2013, 11, 3, 1, 30, tzinfo="US/Eastern", fold=1).fold == 1


def test_utcoffset_dst_and_utctimetuple():
    summer = Arrow(2013, 7, 1, 12, tzinfo="US/Pacific")
    winter = Arrow(2013, 1, 1, 12, tzinfo="US/Pacific")
    assert summer.dst() == datetime.timedelta(hours=1)
    assert winter.dst() == datetime.timedelta(0)
    late = Arrow(2013, 5, 11, 21, 23, 58, tzinfo="US/Pacific")
    tup = late.utctimetuple()
    assert (tup.tm_mday, tup.tm_hour) == (12, 4)
    assert late.timetuple().tm_hour == 21


def test_arithmetic_and_comparison():
    a = Arrow(2013, 5, 11)
    b = Arrow(2013, 5, 12)
    assert a + datetime.timedelta(days=1) == b
    assert isinstance(a + datetime.timedelta(days=1), Arrow)
    assert b - a == datetime.timedelta(days=1)
    assert a < b and b > a and a != b
    assert a == datetime.datetime(2013, 5, 11, tzinfo=datetime.timezone.utc)
    assert sorted([b, a]) == [a, b]
    assert a.to("US/Pacific") == a


def test_is_between_default_is_exclusive():
    start, end = Arrow(2013, 5, 5), Arrow(2013, 5, 7)
    assert Arrow(2013, 5, 6).is_between(start, end) is True
    assert start.is_between(start, end) is False
    assert end.is_between(start, end) is False


def test_is_between_inclusive_bounds():
    start, end = Arrow(2013, 5, 5), Arrow(2013, 5, 7)
    assert start.is_between(start, end, "[]") is True
    assert end.is_between(start, end, "[]") is True
    assert end.is_between(start, end, "[)") is False
    assert end.is_between(start, end, "(]") is True
    assert start.is_between(start, end, "(]") is False


def test_is_between_invalid_bounds_raises():
    start, end = Arrow(2013, 5, 5), Arrow(2013, 5, 7)
    with pytest.raises(ValueError):
        Arrow(2013, 5, 6).is_between(start, end, "][")


def test_clone_is_equal_but_distinct():
    c = A.clone()
    assert c == A
    assert c is not A
    assert isinstance(c, Arrow)


def test_ordinal_roundtrip():
    a = Arrow.fromordinal(737741)
    assert a == Arrow(2020, 11, 12)
    assert a.toordinal() == 737741


def test_alternate_constructors():
    assert Arrow.strptime("2013-05-11 21:23", "%Y-%m-%d %H:%M") == Arrow(
        2013, 5, 11, 21, 23
    )
    assert Arrow.fromdate(datetime.date(2013, 5, 11)) == Arrow(2013, 5, 11)
    dt = datetime.datetime(2013, 5, 11, 12)
    a = Arrow.fromdatetime(dt, tzinfo="US/Pacific")
    assert a.hour == 12
    assert a.utcoffset() == datetime.timedelta(hours=-7)
    assert Arrow.utcfromtimestamp(0) == Arrow(1970, 1, 1)
    assert Arrow.fromtimestamp(0, tzinfo="UTC") == Arrow(1970, 1, 1)


def test_parser_parse_iso_and_parse():
    parser = DateTimeParser()
    assert parser.parse_iso("2013-02-03T04:05:06.789") == datetime.datetime(
        2013, 2, 3, 4, 5, 6, 789000
    )
    assert parser.parse("2013-02-03", "YYYY-MM-DD") == datetime.datetime(2013, 2, 3)
    assert parser.parse(
        "03/02/2013", ["YYYY-MM-DD", "DD/MM/YYYY"]
    ) == datetime.datetime(2013, 2, 3)


def test_parser_errors_and_hierarchy():
    parser = DateTimeParser()
    assert issubclass(ParserError, ValueError)
    assert issubclass(ParserMatchError, ParserError)
    with pytest.raises(ParserError):
        parser.parse_iso("not a date")
    with pytest.raises(ParserError):
        parser.parse("2013-02-03", "MM/DD/YYYY")


def test_tzinfo_parser():
    assert TzinfoParser.parse("+02:00").utcoffset(None) == datetime.timedelta(hours=2)
    assert TzinfoParser.parse("UTC").utcoffset(None) == datetime.timedelta(0)
    with pytest.raises(ParserError):
        TzinfoParser.parse("definitely/not_a_zone")


def test_datetime_formatter():
    dt = datetime.datetime(2013, 5, 11, 21, 23, 58)
    assert DateTimeFormatter().format(dt, "YYYY-MM-DD HH:mm:ss") == "2013-05-11 21:23:58"
    assert DateTimeFormatter("fr").format(dt, "MMMM") == "mai"
