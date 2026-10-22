import datetime
import time

import pytest
from freezegun import freeze_time

import arrow
from arrow.factory import ArrowFactory
from arrow.parser import ParserError


def test_get_iso_string_with_offset():
    a = arrow.get("2013-05-11T21:23:58.970460+07:00")
    assert (a.year, a.month, a.day) == (2013, 5, 11)
    assert (a.hour, a.minute, a.second, a.microsecond) == (21, 23, 58, 970460)
    assert a.utcoffset() == datetime.timedelta(hours=7)


def test_get_iso_date_only():
    a = arrow.get("2013-05-11")
    assert (a.year, a.month, a.day, a.hour) == (2013, 5, 11, 0)
    assert a.utcoffset() == datetime.timedelta(0)


def test_get_iso_year_month():
    a = arrow.get("2013-05")
    assert (a.year, a.month, a.day) == (2013, 5, 1)


def test_get_iso_week_date():
    a = arrow.get("2013-W19-6")
    assert a.date() == datetime.date(2013, 5, 11)


def test_get_from_integers():
    a = arrow.get(2013, 5, 11, 21, 23, 58)
    assert a.isoformat() == "2013-05-11T21:23:58+00:00"


def test_get_naive_datetime_assumes_utc():
    a = arrow.get(datetime.datetime(2013, 5, 11, 12, 30))
    assert a.hour == 12
    assert a.utcoffset() == datetime.timedelta(0)


def test_get_aware_datetime_keeps_offset():
    tz = datetime.timezone(datetime.timedelta(hours=-3))
    a = arrow.get(datetime.datetime(2013, 5, 11, 12, 30, tzinfo=tz))
    assert a.hour == 12
    assert a.utcoffset() == datetime.timedelta(hours=-3)


def test_get_from_date():
    a = arrow.get(datetime.date(2013, 5, 11))
    assert a.datetime.date() == datetime.date(2013, 5, 11)
    assert (a.hour, a.minute, a.second) == (0, 0, 0)
    assert a.utcoffset() == datetime.timedelta(0)


def test_get_from_int_timestamp():
    a = arrow.get(1368303838)
    assert a.isoformat() == "2013-05-11T20:23:58+00:00"


def test_get_from_float_timestamp():
    a = arrow.get(1368303838.5)
    assert a.second == 58
    assert a.microsecond == 500000


def test_get_from_struct_time():
    a = arrow.get(time.gmtime(0))
    assert a == arrow.Arrow(1970, 1, 1)


def test_get_from_arrow_returns_equal():
    a = arrow.Arrow(2013, 5, 11, 12)
    b = arrow.get(a)
    assert a == b


def test_get_with_format_string():
    a = arrow.get("2013-05-05 12:30:45", "YYYY-MM-DD HH:mm:ss")
    assert a == arrow.Arrow(2013, 5, 5, 12, 30, 45)


def test_get_with_format_list():
    a = arrow.get("2013-05-05", ["MM/DD/YYYY", "YYYY-MM-DD"])
    assert a.date() == datetime.date(2013, 5, 5)


def test_get_datetime_and_tz_expression():
    a = arrow.get(datetime.datetime(2013, 5, 5, 12, 30), "US/Pacific")
    assert a.hour == 12
    assert a.utcoffset() == datetime.timedelta(hours=-7)


def test_get_tzinfo_keyword_on_string():
    a = arrow.get("2013-05-05T12:30:45", tzinfo="Europe/Paris")
    assert a.hour == 12
    assert a.utcoffset() == datetime.timedelta(hours=2)


def test_get_invalid_string_raises_parser_error():
    with pytest.raises(ParserError):
        arrow.get("this is not a date")


def test_get_none_raises_type_error():
    with pytest.raises(TypeError):
        arrow.get(None)


def test_get_no_args_and_utcnow_frozen():
    with freeze_time("2013-05-08 05:19:07"):
        expected = arrow.Arrow(2013, 5, 8, 5, 19, 7)
        assert arrow.get() == expected
        assert arrow.utcnow() == expected
        shifted = arrow.now("+02:00")
        assert shifted == expected
        assert shifted.utcoffset() == datetime.timedelta(hours=2)


def test_factory_function_returns_subclass_instances():
    class MyArrow(arrow.Arrow):
        pass

    factory = arrow.factory(MyArrow)
    assert isinstance(factory, ArrowFactory)
    assert isinstance(factory.get(2013, 5, 5), MyArrow)
    assert isinstance(factory.utcnow(), MyArrow)


def test_arrow_factory_default_type():
    factory = ArrowFactory()
    a = factory.get("2013-05-05T12:30:45+00:00")
    assert isinstance(a, arrow.Arrow)
    assert a.hour == 12


def test_arrow_constructor_defaults_to_utc():
    a = arrow.Arrow(2013, 5, 5)
    assert (a.hour, a.minute, a.second, a.microsecond) == (0, 0, 0, 0)
    assert a.utcoffset() == datetime.timedelta(0)
    assert a.fold == 0


def test_arrow_constructor_invalid_month():
    with pytest.raises(ValueError):
        arrow.Arrow(2013, 13, 1)


def test_arrow_constructor_with_tz_string():
    a = arrow.Arrow(2013, 7, 1, 12, tzinfo="US/Pacific")
    assert a.utcoffset() == datetime.timedelta(hours=-7)
    assert a.hour == 12
