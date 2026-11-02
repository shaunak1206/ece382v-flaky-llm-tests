import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

import arrow
from arrow import parser
from arrow.arrow import Arrow
from arrow.factory import ArrowFactory


def test_get_no_args_returns_utc_now():
    with freeze_time("2020-01-01 12:00:00"):
        arw = arrow.get()
        assert isinstance(arw, Arrow)
        assert arw == Arrow(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_utcnow_returns_frozen_utc_time():
    with freeze_time("2020-01-01 12:00:00", tz_offset=0):
        arw = arrow.utcnow()
        assert arw.tzinfo == timezone.utc
        assert arw.year == 2020 and arw.month == 1 and arw.day == 1


def test_now_defaults_to_local_timezone():
    with freeze_time("2020-01-01 12:00:00"):
        arw = arrow.now()
        assert isinstance(arw, Arrow)
        assert arw.tzinfo is not None


def test_now_with_timezone_string():
    with freeze_time("2020-01-01 12:00:00"):
        arw = arrow.now("UTC")
        assert arw.tzinfo == timezone.utc


def test_factory_returns_custom_factory():
    f = arrow.factory(Arrow)
    assert isinstance(f, ArrowFactory)
    assert f.type is Arrow


def test_get_from_arrow_returns_copy():
    original = Arrow(2021, 5, 4, 3, 2, 1, tzinfo=timezone.utc)
    copied = arrow.get(original)
    assert copied is not original
    assert copied == original


def test_get_from_int_timestamp_utc_by_default():
    arw = arrow.get(0)
    assert arw == Arrow(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_get_from_float_timestamp():
    arw = arrow.get(1367992474.293378)
    assert arw.year == 2013
    assert arw.month == 5
    assert arw.day == 8
    assert arw.microsecond == 293378


def test_get_from_decimal_timestamp():
    from decimal import Decimal

    arw = arrow.get(Decimal("1.5"))
    assert arw == Arrow(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)


def test_get_from_none_raises_type_error():
    with pytest.raises(TypeError):
        arrow.get(None)


def test_get_from_datetime_naive_defaults_to_utc():
    dt = datetime(2020, 1, 2, 3, 4, 5)
    arw = arrow.get(dt)
    assert arw.tzinfo == timezone.utc
    assert arw.datetime == dt.replace(tzinfo=timezone.utc)


def test_get_from_datetime_with_tz_replacement():
    dt = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    arw = arrow.get(dt, "US/Pacific")
    assert arw.tzinfo is not None
    assert arw.hour == 3
    assert arw.utcoffset() != timedelta(0)


def test_get_from_date_defaults_to_utc_midnight():
    arw = arrow.get(date(2020, 1, 2))
    assert arw == Arrow(2020, 1, 2, 0, 0, 0, tzinfo=timezone.utc)


def test_get_from_date_with_timezone_replacement():
    arw = arrow.get(date(2020, 1, 2), "UTC")
    assert arw.tzinfo == timezone.utc
    assert arw.hour == 0


def test_get_from_struct_time():
    st = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timetuple()
    arw = arrow.get(st)
    assert arw == Arrow(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_get_from_iso_calendar_tuple():
    arw = arrow.get((2020, 1, 3))
    assert arw.year == 2020
    assert arw.month == 1
    assert arw.day == 1


def test_get_from_string_iso_parses():
    arw = arrow.get("2020-01-02T03:04:05")
    assert arw == Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_from_string_basic_iso_parses():
    arw = arrow.get("20200102T030405")
    assert arw == Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_from_string_with_format_parses():
    arw = arrow.get("2020-01-02 03:04:05", "YYYY-MM-DD HH:mm:ss")
    assert arw == Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_from_string_with_format_list_parses_second_format():
    arw = arrow.get("2020-01-02 03:04:05", ["MM/DD/YYYY", "YYYY-MM-DD HH:mm:ss"])
    assert arw == Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_from_datetime_and_timezone_string_overload():
    dt = datetime(2020, 1, 2, 3, 4, 5)
    arw = arrow.get(dt, "UTC")
    assert arw.tzinfo == timezone.utc


def test_get_from_tzinfo_current_time():
    with freeze_time("2020-01-01 12:00:00"):
        arw = arrow.get(timezone.utc)
        assert arw.tzinfo == timezone.utc
        assert arw == Arrow(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_get_with_tzinfo_kwarg_on_now():
    with freeze_time("2020-01-01 12:00:00"):
        arw = arrow.get(tzinfo="UTC")
        assert arw.tzinfo == timezone.utc


def test_get_with_normalize_whitespace():
    arw = arrow.get("2020-01-02   03:04:05", "YYYY-MM-DD HH:mm:ss", normalize_whitespace=True)
    assert arw == Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_get_string_with_bad_format_raises():
    with pytest.raises(parser.ParserError):
        arrow.get("not-a-date", "YYYY-MM-DD")


def test_arrow_fromtimestamp_invalid_raises():
    with pytest.raises(ValueError):
        Arrow.fromtimestamp("abc")


def test_arrow_utcfromtimestamp_zero():
    arw = Arrow.utcfromtimestamp(0)
    assert arw == Arrow(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_arrow_fromdatetime_preserves_timezone_by_default():
    dt = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    arw = Arrow.fromdatetime(dt)
    assert arw.tzinfo.utcoffset(None) == timedelta(hours=2)


def test_arrow_fromdate_sets_midnight():
    arw = Arrow.fromdate(date(2020, 1, 2))
    assert arw.hour == 0 and arw.minute == 0 and arw.second == 0


def test_arrow_strptime_parses():
    arw = Arrow.strptime("20-01-2020 15:49:10", "%d-%m-%Y %H:%M:%S")
    assert arw.year == 2020 and arw.month == 1 and arw.day == 20


def test_arrow_fromordinal():
    arw = Arrow.fromordinal(date(2020, 1, 1).toordinal())
    assert arw == Arrow(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_repr_and_str():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert repr(arw) == "<Arrow [2020-01-02T03:04:05+00:00]>"
    assert str(arw) == "2020-01-02T03:04:05+00:00"


def test_format_uses_arrow_tokens():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert arw.format("YYYY-MM-DD HH:mm:ss ZZ") == "2020-01-02 03:04:05 +00:00"


def test_format_with_literal_brackets():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert arw.format("[X] YYYY") == "X 2020"


def test_isoformat_and_for_json():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert arw.isoformat() == "2020-01-02T03:04:05+00:00"
    assert arw.for_json() == arw.isoformat()


def test_date_time_timetz_and_timetuple():
    arw = Arrow(2020, 1, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert arw.date() == date(2020, 1, 2)
    assert arw.time().hour == 3
    assert arw.timetz().tzinfo == timezone.utc
    assert arw.timetuple().tm_year == 2020


def test_naive_property_removes_tzinfo():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert arw.naive.tzinfo is None


def test_timestamp_properties():
    arw = Arrow(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    assert math.isclose(arw.timestamp(), 1.5)
    assert arw.int_timestamp == 1
    assert arw.float_timestamp == arw.timestamp()


def test_clone_creates_equal_distinct_object():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    cloned = arw.clone()
    assert cloned == arw
    assert cloned is not arw


def test_replace_absolute_fields():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    replaced = arw.replace(year=2021, month=12)
    assert replaced.year == 2021 and replaced.month == 12 and replaced.day == 2


def test_replace_tzinfo_changes_timezone_without_conversion():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    replaced = arw.replace(tzinfo="US/Pacific")
    assert replaced.tzinfo is not None
    assert replaced.hour == 3


def test_replace_fold():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    replaced = arw.replace(fold=1)
    assert replaced.fold == 1


def test_replace_week_raises():
    arw = Arrow(2020, 1, 2, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        arw.replace(week=2)


def test_shift_years_months_and_quarters():
    arw = Arrow(2020, 1, 31, tzinfo=timezone.utc)
    shifted = arw.shift(months=1)
    assert shifted.year == 2020
    assert shifted.month in (2, 3)


def test_shift_weekday_moves_forward():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)  # Wednesday
    shifted = arw.shift(weekday=0)  # Monday
    assert shifted.weekday() == 0
    assert shifted >= arw


def test_shift_invalid_frame_raises():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        arw.shift(blah=1)


def test_to_timezone_converts_time():
    arw = Arrow(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    converted = arw.to("US/Pacific")
    assert converted.hour == 4
    assert converted.tzinfo is not None


def test_arithmetic_add_and_subtract_timedelta():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    assert arw + timedelta(days=1) == Arrow(2020, 1, 2, tzinfo=timezone.utc)
    assert arw - timedelta(days=1) == Arrow(2019, 12, 31, tzinfo=timezone.utc)


def test_arithmetic_add_relativedelta():
    from dateutil.relativedelta import relativedelta

    arw = Arrow(2020, 1, 31, tzinfo=timezone.utc)
    shifted = arw + relativedelta(months=1)
    assert isinstance(shifted, Arrow)


def test_subtract_arrow_returns_timedelta():
    a = Arrow(2020, 1, 2, tzinfo=timezone.utc)
    b = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    assert a - b == timedelta(days=1)


def test_rsub_datetime_returns_timedelta():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    dt = datetime(2020, 1, 2, tzinfo=timezone.utc)
    assert dt - arw == timedelta(days=1)


def test_comparisons_with_arrow_and_datetime():
    a = Arrow(2020, 1, 2, tzinfo=timezone.utc)
    b = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    assert a > b
    assert a >= b
    assert b < a
    assert b <= a
    assert a == datetime(2020, 1, 2, tzinfo=timezone.utc)


def test_week_and_quarter_properties():
    arw = Arrow(2020, 5, 15, tzinfo=timezone.utc)
    assert arw.quarter == 2
    assert isinstance(arw.week, int)


def test_ambiguous_and_imaginary_are_boolean():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    assert isinstance(arw.ambiguous, bool)
    assert isinstance(arw.imaginary, bool)


def test_is_between_default_exclusive():
    start = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    end = Arrow(2020, 1, 3, tzinfo=timezone.utc)
    assert Arrow(2020, 1, 2, tzinfo=timezone.utc).is_between(start, end)
    assert not start.is_between(start, end)


def test_is_between_inclusive_bounds():
    start = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    end = Arrow(2020, 1, 3, tzinfo=timezone.utc)
    assert start.is_between(start, end, "[]")
    assert end.is_between(start, end, "[]")


def test_is_between_invalid_bounds_raises():
    start = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    end = Arrow(2020, 1, 3, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        start.is_between(start, end, "bad")


def test_range_by_hour_includes_expected_points():
    start = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2020, 1, 1, 14, 0, tzinfo=timezone.utc)
    values = list(Arrow.range("hour", start, end))
    assert [v.hour for v in values] == [12, 13, 14]


def test_range_with_limit():
    start = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    values = list(Arrow.range("hour", start, limit=2))
    assert len(values) == 2


def test_range_without_end_or_limit_raises():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        list(Arrow.range("day", start))


def test_range_invalid_frame_raises():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        list(Arrow.range("century", start, start))


def test_span_day_default_bounds():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    floor, ceil = arw.span("day")
    assert floor == Arrow(2020, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert ceil.hour == 23 and ceil.minute == 59


def test_floor_and_ceil_helpers():
    arw = Arrow(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert arw.floor("hour").minute == 0
    assert arw.ceil("hour").minute == 59


def test_span_week_with_week_start():
    arw = Arrow(2021, 2, 24, tzinfo=timezone.utc)  # Wednesday
    floor, ceil = arw.span("week", week_start=1)
    assert floor.isoweekday() == 1
    assert ceil.isoweekday() == 7


def test_span_invalid_week_start_raises():
    arw = Arrow(2021, 2, 24, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        arw.span("week", week_start=0)


def test_span_range_day_returns_spans():
    start = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2020, 1, 3, 12, 0, tzinfo=timezone.utc)
    spans = list(Arrow.span_range("day", start, end))
    assert len(spans) >= 2
    assert spans[0][0].hour == 0
    assert spans[0][1].hour == 23


def test_interval_groups_spans():
    start = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2020, 1, 1, 18, 0, tzinfo=timezone.utc)
    intervals = list(Arrow.interval("hour", start, end, interval=2))
    assert len(intervals) >= 2
    assert intervals[0][0].minute == 0


def test_interval_invalid_interval_raises():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        list(Arrow.interval("hour", start, start, interval=0))


def test_utcfromtimestamp_and_fromtimestamp_with_timezone():
    arw = Arrow.fromtimestamp(0, tzinfo="UTC")
    assert arw == Arrow.utcfromtimestamp(0)


def test_get_tzinfo_parses_valid_string():
    tz = Arrow._get_tzinfo("UTC")
    assert tz == timezone.utc


def test_get_tzinfo_invalid_raises():
    with pytest.raises(ValueError):
        Arrow._get_tzinfo("not-a-zone")


def test_get_datetime_accepts_timestamp_and_arrow():
    arw = Arrow(2020, 1, 1, tzinfo=timezone.utc)
    assert Arrow._get_datetime(arw) == arw.datetime
    assert Arrow._get_datetime(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_get_frames_supports_plural_and_quarter():
    assert Arrow._get_frames("day") == ("day", "days", 1)
    assert Arrow._get_frames("days") == ("day", "days", 1)
    assert Arrow._get_frames("quarter") == ("quarter", "months", 3)


def test_get_frames_invalid_raises():
    with pytest.raises(ValueError):
        Arrow._get_frames("decade")


def test_parser_tzinfo_parser_utc_local_and_offset():
    assert parser.TzinfoParser.parse("UTC") == timezone.utc
    assert parser.TzinfoParser.parse("Z") == timezone.utc
    tz = parser.TzinfoParser.parse("+02:00")
    assert tz.utcoffset(None) == timedelta(hours=2)


def test_parser_tzinfo_parser_invalid_raises():
    with pytest.raises(parser.ParserError):
        parser.TzinfoParser.parse("invalid-zone")


def test_parser_parse_iso_and_normalize_whitespace():
    p = parser.DateTimeParser()
    dt = p.parse_iso("2020-01-02T03:04:05")
    assert dt == datetime(2020, 1, 2, 3, 4, 5)
    dt2 = p.parse_iso(" 2020-01-02   03:04:05 ", normalize_whitespace=True)
    assert dt2 == datetime(2020, 1, 2, 3, 4, 5)


def test_parser_parse_format_with_list_tries_multiple():
    p = parser.DateTimeParser()
    dt = p.parse("2020-01-02 03:04:05", ["MM/DD/YYYY", "YYYY-MM-DD HH:mm:ss"])
    assert dt == datetime(2020, 1, 2, 3, 4, 5)


def test_parser_parse_invalid_format_raises():
    p = parser.DateTimeParser()
    with pytest.raises(parser.ParserMatchError):
        p.parse("2020-01-02", "YYYY/MM/DD")


def test_parser_build_datetime_with_weekdate():
    p = parser.DateTimeParser()
    parts = {"weekdate": ("2020", "01", "1")}
    dt = p._build_datetime(parts)
    assert dt.year == 2019 or dt.year == 2020


def test_parser_build_datetime_with_day_of_year_requires_year():
    p = parser.DateTimeParser()
    with pytest.raises(parser.ParserError):
        p._build_datetime({"day_of_year": 1})


def test_parser_build_datetime_midnight_24_is_next_day():
    p = parser.DateTimeParser()
    dt = p._build_datetime({"year": 2020, "month": 1, "day": 1, "hour": 24})
    assert dt == datetime(2020, 1, 2, 0, 0)


def test_parser_generate_choice_re_matches_one_of_values():
    re_obj = parser.DateTimeParser._generate_choice_re(["foo", "bar"])
    assert re_obj.search("bar")
    assert not re_obj.search("baz")
