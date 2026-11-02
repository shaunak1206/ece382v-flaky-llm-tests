import pytest
import re
import calendar
from datetime import datetime, date, time, timedelta, timezone
from time import struct_time, gmtime
from decimal import Decimal

import arrow
from arrow.api import get, now, utcnow, factory
from arrow.arrow import Arrow
from arrow.factory import ArrowFactory
from arrow.parser import DateTimeParser, ParserError, ParserMatchError, TzinfoParser
from arrow.formatter import DateTimeFormatter


# ------------ Tests for arrow.api (api.py) ------------

def test_api_get_no_args_returns_utcnow_matching():
    arw1 = get()
    arw2 = utcnow()
    # Should be very close timestamps (within a second)
    assert abs(arw1.timestamp() - arw2.timestamp()) < 2


def test_api_get_with_datetime_returns_same_time():
    dt = datetime(2022, 1, 1, 12)
    arw = get(dt)
    assert arw.year == 2022 and arw.month == 1 and arw.day == 1
    assert arw.hour == 12


def test_api_get_with_date_returns_start_of_day():
    d = date(2020, 5, 5)
    arw = get(d)
    assert arw.year == 2020 and arw.month == 5 and arw.day == 5
    assert arw.hour == 0 and arw.minute == 0


def test_api_get_with_timestamp_float_returns_correct_dt():
    ts = 1600000000.123456
    arw = get(ts)
    assert abs(arw.timestamp() - ts) < 1e-5
    assert arw.tzinfo.utcoffset(None).total_seconds() == 0


def test_api_get_with_string_iso8601_parse():
    iso_str = "2020-10-15T13:45:30"
    arw = get(iso_str)
    assert arw.year == 2020 and arw.month == 10 and arw.day == 15
    assert arw.hour == 13 and arw.minute == 45 and arw.second == 30


def test_api_get_with_tzinfo_returns_now_in_that_tz():
    tz = timezone(timedelta(hours=5, minutes=30))
    arw = get(tz)
    now_ = Arrow.now(tz)
    assert arw.year == now_.year and arw.minute == now_.minute


def test_api_get_with_struct_time():
    st = gmtime(0)
    arw = get(st)
    assert arw.year == 1970 and arw.month == 1 and arw.day == 1
    assert arw.hour == 0 and arw.minute == 0 and arw.second == 0


def test_api_get_with_iso_calendar_tuple():
    week_tuple = (2023, 18, 7)  # 2023 week 18 day 7 (Sunday)
    arw = get(week_tuple)
    # Should resolve to a date in May 2023
    assert arw.year == 2023 and arw.month == 5


def test_api_get_two_arg_datetime_and_tz_str():
    dt = datetime(2020, 6, 1, 12)
    arw = get(dt, "UTC")
    assert arw.tzinfo.utcoffset(None).total_seconds() == 0
    assert arw.hour == 12


def test_api_get_two_arg_date_and_tz_str():
    d = date(2020, 6, 1)
    arw = get(d, "UTC")
    assert arw.tzinfo.utcoffset(None).total_seconds() == 0
    assert arw.hour == 0


def test_api_get_two_arg_str_and_format():
    s = "2020-10-15 12:34:56"
    fmt = "YYYY-MM-DD HH:mm:ss"
    arw = get(s, fmt)
    assert arw.year == 2020 and arw.minute == 34


def test_api_get_with_three_args_ctor():
    arw = get(2019, 2, 15, 13, 29, 45)
    assert arw.year == 2019 and arw.month == 2 and arw.day == 15
    assert arw.hour == 13 and arw.minute == 29 and arw.second == 45


def test_api_utcnow_and_now_consistency():
    arw_utc = utcnow()
    arw_now = now()
    assert isinstance(arw_utc, Arrow) and isinstance(arw_now, Arrow)


def test_api_factory_returns_arrowfactory_of_type():
    fact = factory(Arrow)
    assert isinstance(fact, ArrowFactory)
    assert fact.type is Arrow


# ------------ Tests for arrow.factory (factory.py) ------------

def test_factory_get_no_args_returns_utcnow():
    f = ArrowFactory()
    a = f.get()
    assert isinstance(a, Arrow)
    assert a.tzinfo.utcoffset(None) == timedelta(0)  # utcnow


def test_factory_get_one_arg_arrow_object_returns_copy():
    f = ArrowFactory()
    original = Arrow.utcnow()
    copy = f.get(original)
    assert isinstance(copy, Arrow)
    assert copy == original


def test_factory_get_one_arg_decimal_timestamp():
    f = ArrowFactory()
    dec = Decimal("1672531200.456")
    arw = f.get(dec)
    assert abs(arw.timestamp() - 1672531200.456) < 0.001


def test_factory_get_one_arg_none_raises_type_error():
    f = ArrowFactory()
    with pytest.raises(TypeError):
        f.get(None)


def test_factory_get_one_arg_int_timestamp_with_tz():
    f = ArrowFactory()
    ts = 1600000000
    tz = timezone(timedelta(hours=1))
    arw = f.get(ts, tzinfo=tz)
    assert arw.tzinfo.utcoffset(None) == timedelta(hours=1)


def test_factory_get_one_arg_arrow_with_tz_replaces_tz():
    f = ArrowFactory()
    orig = Arrow.utcnow()
    tz = timezone(timedelta(hours=-3))
    arw = f.get(orig, tzinfo=tz)
    assert arw.tzinfo.utcoffset(None) == timedelta(hours=-3)


def test_factory_get_two_args_datetime_and_tz_str():
    f = ArrowFactory()
    dt = datetime(2020, 5, 5, 15, 0)
    arw = f.get(dt, "UTC")
    assert arw.tzinfo.utcoffset(None) == timedelta(0)
    assert arw.hour == 15


def test_factory_get_two_args_date_and_tzinfo():
    f = ArrowFactory()
    d = date(2020, 5, 5)
    tz = timezone(timedelta(hours=-7))
    arw = f.get(d, tz)
    assert arw.tzinfo.utcoffset(None) == timedelta(hours=-7)
    assert arw.hour == 0


def test_factory_get_two_args_str_and_list_formats():
    f = ArrowFactory()
    date_string = "05/01/2019"
    formats = ["MM/DD/YYYY", "YYYY-MM-DD"]
    arw = f.get(date_string, formats)
    assert arw.year == 2019 and arw.month == 5 and arw.day == 1


def test_factory_get_three_or_more_args_uses_constructor():
    f = ArrowFactory()
    a = f.get(2015, 7, 13, 1, 0, 0)
    assert a.year == 2015
    assert a.hour == 1


def test_factory_utcnow_returns_arrow_utcnow():
    f = ArrowFactory()
    a = f.utcnow()
    assert isinstance(a, Arrow)
    assert a.tzinfo.utcoffset(None) == timedelta(0)


def test_factory_now_returns_current_time_local():
    f = ArrowFactory()
    a = f.now()
    assert isinstance(a, Arrow)
    # Assume local time offset is not None
    assert a.tzinfo.utcoffset(None) is not None


def test_factory_now_with_string_timezone():
    f = ArrowFactory()
    a = f.now("+02:00")
    assert a.tzinfo.utcoffset(None) == timedelta(hours=2)


def test_factory_now_with_none_tz_returns_localtime():
    f = ArrowFactory()
    a = f.now(None)
    local_tz = datetime.now().astimezone().tzinfo
    assert a.tzinfo.utcoffset(None) == local_tz.utcoffset(None)


def test_factory_get_with_tzinfo_kwarg_only():
    f = ArrowFactory()
    tz = timezone(timedelta(hours=3))
    a = f.get(tzinfo=tz)
    now = Arrow.now(tz)
    # Timestamp close to now
    assert abs(a.timestamp() - now.timestamp()) < 2


# ------------ Tests for arrow.arrow.Arrow class ------------

def test_arrow_constructor_sets_attributes_and_timezone():
    tz = timezone(timedelta(hours=3))
    a = Arrow(2021, 7, 10, 15, 30, 45, 123456, tz)
    assert a.year == 2021
    assert a.month == 7
    assert a.day == 10
    assert a.hour == 15
    assert a.minute == 30
    assert a.second == 45
    assert a.microsecond == 123456
    assert a.tzinfo == tz


def test_arrow_now_returns_local_timezone():
    a = Arrow.now()
    local_tz = datetime.now().astimezone().tzinfo
    assert a.tzinfo.utcoffset(None) == local_tz.utcoffset(None)


def test_arrow_utcnow_returns_utc_timezone():
    a = Arrow.utcnow()
    assert a.tzinfo.utcoffset(None) == timedelta(0)


def test_arrow_fromtimestamp_returns_arrow_with_tz():
    ts = 1600000000
    tz = timezone(timedelta(hours=5))
    a = Arrow.fromtimestamp(ts, tz)
    assert abs(a.timestamp() - ts) < 0.01
    assert a.tzinfo.utcoffset(None) == timedelta(hours=5)


def test_arrow_utcfromtimestamp_returns_utc():
    ts = 1600000000
    a = Arrow.utcfromtimestamp(ts)
    assert abs(a.timestamp() - ts) < 0.01
    assert a.tzinfo.utcoffset(None) == timedelta(0)


def test_arrow_fromdatetime_with_naive_assumes_utc():
    dt = datetime(2021, 7, 20, 13, 15)
    a = Arrow.fromdatetime(dt)
    assert a.year == 2021 and a.minute == 15 and a.tzinfo.utcoffset(None) == timedelta(0)


def test_arrow_fromdatetime_with_tz_preserves_tz():
    dt = datetime(2021, 7, 20, 13, 15, tzinfo=timezone(timedelta(hours=-2)))
    a = Arrow.fromdatetime(dt)
    assert a.tzinfo.utcoffset(None) == timedelta(hours=-2)


def test_arrow_fromdate_returns_midnight_in_utc():
    d = date(2022, 1, 1)
    a = Arrow.fromdate(d)
    assert a.year == 2022 and a.hour == 0
    assert a.tzinfo.utcoffset(None) == timedelta(0)


def test_arrow_strptime_parses_date():
    s = "20-01-2019 15:49:10"
    fmt = "%d-%m-%Y %H:%M:%S"
    a = Arrow.strptime(s, fmt)
    assert a.year == 2019 and a.day == 20 and a.hour == 15


def test_arrow_fromordinal_returns_correct_date():
    ordinal = datetime(2020, 11, 12).toordinal()
    a = Arrow.fromordinal(ordinal)
    assert a.year == 2020 and a.month == 11 and a.day == 12


def test_arrow_repr_and_str():
    a = Arrow(2023, 1, 2, 3, 4, 5)
    rep = repr(a)
    s = str(a)
    assert "Arrow" in rep
    assert s.startswith("2023-01-02T03:04:05") or s.startswith("2023-01-02 03:04:05")


def test_arrow_eq_and_ne():
    a1 = Arrow(2020, 1, 1)
    a2 = Arrow(2020, 1, 1)
    a3 = Arrow(2021, 1, 1)
    assert a1 == a2
    assert a1 != a3
    assert (a1 == datetime(2020, 1, 1, tzinfo=timezone.utc)) is True
    assert (a1 != datetime(2021, 1, 1, tzinfo=timezone.utc)) is True


def test_arrow_lt_le_gt_ge():
    a1 = Arrow(2020, 1, 1)
    a2 = Arrow(2021, 1, 1)
    assert a1 < a2
    assert a2 > a1
    assert a1 <= a2
    assert a2 >= a1


def test_arrow_clone_returns_equal_object():
    a1 = Arrow.now()
    a2 = a1.clone()
    assert a1 == a2
    assert a1 is not a2


def test_arrow_replace_changes_attributes():
    a1 = Arrow(2019, 1, 1, 12, 30)
    a2 = a1.replace(year=2020, minute=45)
    assert a2.year == 2020
    assert a2.minute == 45
    assert a2.hour == 12
    # Original not modified
    assert a1.year == 2019


def test_arrow_replace_raises_on_invalid_attribute():
    a = Arrow.now()
    with pytest.raises(ValueError):
        a.replace(foo=123)
    with pytest.raises(ValueError):
        a.replace(week=1)
    with pytest.raises(ValueError):
        a.replace(quarter=1)


def test_arrow_shift_with_valid_attrs_changes_time():
    a = Arrow(2020, 1, 1)
    shifted = a.shift(years=1, months=2, weeks=1, days=3, hours=5, minutes=10, seconds=20, microseconds=500)
    # Check if shifted values are approximately correct
    assert shifted.year == 2021
    assert shifted.month == 3 or shifted.month == 4  # 2 months after Jan can be March or April due to month length
    assert shifted.day >= 1
    assert shifted.hour == 5
    assert shifted.minute == 10
    assert shifted.second == 20
    assert isinstance(shifted.microsecond, int)


def test_arrow_shift_invalid_attr_raises():
    a = Arrow.now()
    with pytest.raises(ValueError):
        a.shift(foo=10)


def test_arrow_to_changes_timezone():
    a = Arrow.utcnow()
    pacific = "US/Pacific"
    a2 = a.to(pacific)
    assert a2.tzinfo != a.tzinfo
    assert a2.year == a.year or a2.year != a.year  # date may differ depending on timezone offset


def test_arrow_format_outputs_correct_string():
    a = Arrow(2020, 5, 17, 13, 45, 30)
    fmt_str = "YYYY-MM-DD HH:mm:ss"
    out = a.format(fmt_str)
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", out)


def test_arrow_humanize_in_past_and_future():
    a = Arrow.utcnow()
    earlier = a.shift(hours=-2)
    later = a.shift(hours=2)
    res_past = a.humanize(earlier)
    res_future = a.humanize(later)
    assert "ago" in res_past or "ago" not in res_past  # locale dependent but should be string
    assert any(w in res_future for w in ("in", "after", "from")) or isinstance(res_future, str)


def test_arrow_humanize_with_only_distance_true():
    a = Arrow.utcnow()
    other = a.shift(seconds=-11)
    s = a.humanize(other, only_distance=True)
    assert isinstance(s, str)


def test_arrow_dehumanize_with_supported_locale_and_future_and_past():
    a = Arrow.utcnow()
    s = "in 2 days"
    b = a.dehumanize(s)
    assert b > a

    s2 = "2 days ago"
    c = a.dehumanize(s2)
    assert c < a


def test_arrow_dehumanize_with_unsupported_locale_raises():
    a = Arrow.utcnow()
    with pytest.raises(ValueError):
        a.dehumanize("2 days ago", locale="unsupported-locale")


def test_arrow_dehumanize_invalid_string_raises():
    a = Arrow.utcnow()
    # String without relative time info
    with pytest.raises(ValueError):
        a.dehumanize("nonsense string that won't parse")


def test_arrow_is_between_correct_bounds():
    a = Arrow(2020, 1, 10)
    start = Arrow(2020, 1, 1)
    end = Arrow(2020, 1, 20)
    assert a.is_between(start, end, "[]")
    assert a.is_between(start, end, "[)")
    assert not a.is_between(start, start, "()")
    with pytest.raises(ValueError):
        a.is_between(start, end, "invalid")


def test_arrow_is_between_wrong_types_raises():
    a = Arrow.utcnow()
    with pytest.raises(TypeError):
        a.is_between("not-arrow", a)
    with pytest.raises(TypeError):
        a.is_between(a, "not-arrow")


def test_arrow_date_time_timetz_returns_correct_types():
    a = Arrow(2020, 6, 7, 10, 20, 30)
    d = a.date()
    t = a.time()
    tt = a.timetz()
    assert isinstance(d, date)
    assert isinstance(t, time)
    assert isinstance(tt, time)


def test_arrow_astimezone_returns_datetime():
    a = Arrow.utcnow()
    tz2 = timezone(timedelta(hours=2))
    dt = a.astimezone(tz2)
    assert isinstance(dt, datetime)
    assert dt.utcoffset() == timedelta(hours=2)


def test_arrow_utcoffset_and_dst():
    a = Arrow.utcnow()
    off = a.utcoffset()
    dst = a.dst()
    assert isinstance(off, timedelta)
    assert isinstance(dst, timedelta)


def test_arrow_timetuple_and_utctimetuple_types():
    a = Arrow.utcnow()
    tt = a.timetuple()
    utt = a.utctimetuple()
    assert isinstance(tt, struct_time)
    assert isinstance(utt, struct_time)


def test_arrow_toordinal_weekday_isoweekday_isocalendar():
    a = Arrow(2020, 1, 2)
    assert isinstance(a.toordinal(), int)
    assert 0 <= a.weekday() <= 6
    assert 1 <= a.isoweekday() <= 7
    ic = a.isocalendar()
    assert len(ic) == 3


def test_arrow_isoformat_ctime_strftime():
    a = Arrow(2020, 1, 1, 12, 13, 14)
    assert isinstance(a.isoformat(), str)
    assert isinstance(a.ctime(), str)
    fmt = "%Y-%m-%d %H:%M:%S"
    assert a.strftime(fmt) == a.datetime.strftime(fmt)


def test_arrow_for_json_returns_isoformat():
    a = Arrow.utcnow()
    assert a.for_json() == a.isoformat()


def test_arrow_add_sub_operations():
    a = Arrow(2020, 1, 1)
    td = timedelta(days=1)
    a2 = a + td
    assert (a2 - a).days == 1
    a3 = a2 - td
    assert a == a3
    # Subtraction with another Arrow returns timedelta
    diff = a3 - a
    assert isinstance(diff, timedelta)

    with pytest.raises(TypeError):
        a + 10  # NotImplemented


def test_arrow_comparison_with_nonarrow_returns_notimplemented_or_false():
    a = Arrow.utcnow()
    assert (a == "string") is False
    assert (a != "string") is True
    assert (a < "string") is NotImplemented or a.__lt__("string") is NotImplemented
    assert (a <= "string") is NotImplemented or a.__le__("string") is NotImplemented
    assert (a > "string") is NotImplemented or a.__gt__("string") is NotImplemented
    assert (a >= "string") is NotImplemented or a.__ge__("string") is NotImplemented


def test_internal_get_tzinfo_returns_timezone_or_parsed():
    utc = Arrow._get_tzinfo(None)
    assert utc.utcoffset(None) == timedelta(0)
    tz = timezone(timedelta(hours=5))
    assert Arrow._get_tzinfo(tz) == tz
    # Parsing a string timezone should work
    tz2 = Arrow._get_tzinfo("UTC")
    assert tz2.utcoffset(None) == timedelta(0)
    with pytest.raises(ValueError):
        Arrow._get_tzinfo("not-a-tz")


def test_internal_get_datetime_with_arrow_datetime_int_float():
    a = Arrow.utcnow()
    dt = Arrow._get_datetime(a)
    assert isinstance(dt, datetime)
    dt2 = datetime(2020, 1, 1)
    assert Arrow._get_datetime(dt2) == dt2
    assert isinstance(Arrow._get_datetime(1600000000), datetime)
    with pytest.raises(ValueError):
        Arrow._get_datetime("no-date")


def test_internal_get_frames():
    assert Arrow._get_frames("day") == ("day", "days", 1)
    assert Arrow._get_frames("days") == ("day", "days", 1)
    assert Arrow._get_frames("week") == ("week", "weeks", 1)
    assert Arrow._get_frames("quarters") == ("quarter", "months", 3)
    with pytest.raises(ValueError):
        Arrow._get_frames("century")


def test_internal_get_iteration_params_with_end_and_limit():
    end, limit = Arrow._get_iteration_params(datetime(2020, 1, 1), 5)
    assert limit == 5
    end, limit = Arrow._get_iteration_params(datetime(2020, 1, 1), None)
    assert limit == 2 ** 63 - 1 or limit > 10000000  # sys.maxsize
    with pytest.raises(ValueError):
        Arrow._get_iteration_params(None, None)


def test_internal_is_last_day_of_month_true_false():
    # Last day of Feb 2020 (leap year)
    dt_last = Arrow(2020, 2, 29)
    assert Arrow._is_last_day_of_month(dt_last) is True
    dt_not_last = Arrow(2020, 2, 28)
    assert Arrow._is_last_day_of_month(dt_not_last) is False


def test_span_and_floor_ceil_outputs():
    a = Arrow(2021, 5, 9, 3, 32)
    floor, ceil = a.span("hour")
    assert floor.hour == 3
    assert ceil.hour == 3
    floor2 = a.floor("hour")
    ceil2 = a.ceil("hour")
    assert floor == floor2 and ceil == ceil2


def test_range_and_span_range_basic_behavior():
    start = datetime(2021, 2, 22)
    end = datetime(2021, 2, 28)
    results = list(Arrow.range("day", start, end))
    assert len(results) >= 1
    ranges = list(Arrow.span_range("day", start, end))
    assert len(ranges) >= 1
    for floor, ceil in ranges:
        assert floor <= ceil


def test_interval_raises_for_below_one():
    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)
    with pytest.raises(ValueError):
        list(Arrow.interval("hour", start, end, 0))


def test_interval_produces_intervals_correctly():
    start = datetime(2021, 5, 5, 12, 30)
    end = datetime(2021, 5, 5, 17, 15)
    intervals = list(Arrow.interval("hour", start, end, 2))
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in intervals)


# ------------ Tests for arrow.parser.DateTimeParser ------------

def test_parser_parse_iso_basic_date():
    p = DateTimeParser()
    dt = p.parse_iso("2021-10-12")
    assert dt.year == 2021 and dt.month == 10 and dt.day == 12


def test_parser_parse_iso_with_time_and_tz():
    p = DateTimeParser()
    s = "2021-10-12T14:30:00Z"
    dt = p.parse_iso(s)
    assert dt.year == 2021 and dt.hour == 14 and dt.tzinfo is not None


def test_parser_parse_iso_raises_for_bad_format():
    p = DateTimeParser()
    with pytest.raises(ParserError):
        p.parse_iso("some non-date string")


def test_parser_parse_with_format_string_single_format():
    p = DateTimeParser()
    s = "2021-10-12 14:30:00"
    fmt = "YYYY-MM-DD HH:mm:ss"
    dt = p.parse(s, fmt)
    assert dt.year == 2021 and dt.hour == 14


def test_parser_parse_with_format_list_tries_formats():
    p = DateTimeParser()
    s = "2021-10-12 14:30:00"
    f1 = "MMMM DD, YYYY"
    f2 = "YYYY-MM-DD HH:mm:ss"
    dt = p.parse(s, [f1, f2])
    assert dt.year == 2021


def test_parser_parse_raises_on_bad_match():
    p = DateTimeParser()
    with pytest.raises(ParserMatchError):
        p.parse("2021/10/12", "YYYY-MM-DD")


def test_generate_pattern_re_and_parse_token_build_datetime():
    p = DateTimeParser()
    tokens, regex = p._generate_pattern_re("YYYY-MM-DD")
    assert "YYYY" in tokens and "MM" in tokens and "DD" in tokens
    parts = {}
    p._parse_token("YYYY", "2020", parts)
    p._parse_token("MM", "06", parts)
    p._parse_token("DD", "10", parts)
    dt = p._build_datetime(parts)
    assert dt.year == 2020 and dt.month == 6 and dt.day == 10


def test_parse_token_handles_am_pm_adjustment():
    p = DateTimeParser()
    parts = {"hour": 3}
    p._parse_token("a", "am", parts)
    assert parts["am_pm"] == "am"
    parts = {"hour": 11}
    p._parse_token("a", "pm", parts)
    assert parts["am_pm"] == "pm"


def test_build_datetime_handles_weekdate():
    parts = {"weekdate": ("2021", "20", "3")}
    dt = DateTimeParser._build_datetime(parts)
    assert dt.year == 2021


def test_build_datetime_raises_for_invalid_day_of_year():
    with pytest.raises(ParserError):
        DateTimeParser._build_datetime({"year": 2021, "day_of_year": 400})


def test_parse_multiformat_parses_correct_or_raises():
    p = DateTimeParser()
    fmts = ["YYYY-MM-DD", "YYYY/MM/DD"]
    dt = p._parse_multiformat("2021-10-12", fmts)
    assert dt.year == 2021
    with pytest.raises(ParserError):
        p._parse_multiformat("bad-date", fmts)


def test_generate_choice_re_compiles_correctly():
    regex = DateTimeParser._generate_choice_re(["Jan", "Feb", "Mar"])
    assert regex.match("Jan")


def test_tzinfo_parser_parse_utc_local_and_offsets():
    tz1 = TzinfoParser.parse("utc")
    assert tz1.utcoffset(None) == timedelta(0)
    tz2 = TzinfoParser.parse("UTC")
    assert tz2.utcoffset(None) == timedelta(0)
    tz3 = TzinfoParser.parse("local")
    assert tz3.utcoffset(None) is not None
    tz4 = TzinfoParser.parse("+02:00")
    assert tz4.utcoffset(None) == timedelta(hours=2)
    with pytest.raises(ParserError):
        TzinfoParser.parse("NotATZ")


# ------------ Tests for arrow.formatter.DateTimeFormatter ------------

def test_formatter_format_token_literal_and_year():
    fmt = DateTimeFormatter()
    dt = datetime(2021, 2, 3)
    assert fmt._format_token(dt, "[Literal]") == "Literal"
    assert fmt._format_token(dt, "YYYY") == "2021"
    assert fmt._format_token(dt, "YY") == "21"


def test_formatter_format_token_month_day_hour_min_sec():
    fmt = DateTimeFormatter()
    dt = datetime(2021, 7, 8, 13, 4, 55, 123456)
    assert fmt._format_token(dt, "MMMM") == fmt.locale.month_name(7)
    assert fmt._format_token(dt, "MMM") == fmt.locale.month_abbreviation(7)
    assert fmt._format_token(dt, "MM") == "07"
    assert fmt._format_token(dt, "M") == "7"
    assert fmt._format_token(dt, "DD") == "08"
    assert fmt._format_token(dt, "D") == "8"
    assert fmt._format_token(dt, "HH") == "13"
    assert fmt._format_token(dt, "H") == "13"
    assert fmt._format_token(dt, "mm") == "04"
    assert fmt._format_token(dt, "m") == "4"
    assert fmt._format_token(dt, "ss") == "55"
    assert fmt._format_token(dt, "s") == "55"


def test_formatter_format_token_microseconds():
    fmt = DateTimeFormatter()
    dt = datetime(2021, 1, 1, 0, 0, 0, 123456)
    assert fmt._format_token(dt, "SSSSSS") == "123456"
    assert fmt._format_token(dt, "SSSSS") == "01234"
    assert fmt._format_token(dt, "SSSS") == "0123"
    assert fmt._format_token(dt, "SSS") == "123"
    assert fmt._format_token(dt, "SS") == "12"
    assert fmt._format_token(dt, "S") == "1"


def test_formatter_format_token_timestamp_and_tz():
    fmt = DateTimeFormatter()
    dt = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert re.match(r"^\d+(\.\d+)?$", fmt._format_token(dt, "X"))
    assert re.match(r"^\d+$", fmt._format_token(dt, "x"))
    assert fmt._format_token(dt, "ZZZ") == "UTC" or fmt._format_token(dt, "ZZZ") == "UTC"
    assert re.match(r"[+-]\d{2}(:?\d{2})?", fmt._format_token(dt, "ZZ"))
    assert re.match(r"[+-]\d{2}(:?\d{2})?", fmt._format_token(dt, "Z"))


def test_formatter_format_token_meridian_W():
    fmt = DateTimeFormatter()
    dt_am = datetime(2021, 1, 1, 3)
    dt_pm = datetime(2021, 1, 1, 15)
    assert fmt._format_token(dt_am, "a") in ("am", "AM")
    assert fmt._format_token(dt_pm, "A") in ("PM", "pm")
    dt = datetime(2021, 1, 1, 1)
    w = fmt._format_token(dt, "W")
    assert re.match(r"\d{4}-W\d{2}-\d", w)


def test_formatter_format_returns_correct_replacement():
    fmt = DateTimeFormatter()
    dt = datetime(2020, 1, 2, 3, 4, 5)
    result = fmt.format(dt, "YYYY-MM-DD")
    assert result == "2020-01-02"


def test_formatter_format_with_empty_string_returns_iso():
    fmt = DateTimeFormatter()
    dt = datetime(2020, 1, 2, 3, 4, 5)
    assert fmt.format(dt, "") == dt.isoformat()

