import datetime

import pytest

import arrow
from arrow import Arrow


def test_shift_hours_negative():
    a = Arrow(2013, 5, 11, 21, 23, 58)
    assert a.shift(hours=-1) == Arrow(2013, 5, 11, 20, 23, 58)


def test_shift_days_and_weeks():
    a = Arrow(2013, 5, 11)
    assert a.shift(days=2) == Arrow(2013, 5, 13)
    assert a.shift(weeks=1) == Arrow(2013, 5, 18)


def test_shift_month_end_clamps_day():
    assert Arrow(2013, 1, 31).shift(months=1) == Arrow(2013, 2, 28)


def test_shift_month_leap_year():
    assert Arrow(2012, 1, 31).shift(months=1).day == 29


def test_shift_years_negative():
    assert Arrow(2013, 5, 11).shift(years=-1) == Arrow(2012, 5, 11)


def test_shift_multiple_units():
    a = Arrow(2013, 5, 11, 12)
    assert a.shift(days=1, hours=-2) == Arrow(2013, 5, 12, 10)


def test_shift_invalid_unit_raises():
    with pytest.raises(ValueError):
        Arrow(2013, 5, 11).shift(foo=1)


def test_shift_over_dst_gap_resolves_imaginary():
    a = Arrow(2013, 3, 10, 1, 30, tzinfo="US/Eastern")
    shifted = a.shift(hours=1)
    assert shifted.hour == 3
    assert shifted.imaginary is False


def test_replace_absolute_values():
    a = Arrow(2013, 5, 11, 22, 27, 34)
    r = a.replace(year=2014, month=6)
    assert r == Arrow(2014, 6, 11, 22, 27, 34)
    assert a.year == 2013


def test_replace_tzinfo_does_not_convert():
    a = Arrow(2013, 5, 11, 12)
    r = a.replace(tzinfo="US/Pacific")
    assert r.hour == 12
    assert r.utcoffset() == datetime.timedelta(hours=-7)


def test_replace_unknown_attribute_raises():
    with pytest.raises(ValueError):
        Arrow(2013, 5, 11).replace(foo=3)


def test_to_converts_but_keeps_instant():
    utc = Arrow(2013, 5, 11, 20, 23, 58)
    local = utc.to("US/Pacific")
    assert local.hour == 13
    assert local.utcoffset() == datetime.timedelta(hours=-7)
    assert local == utc


def test_span_day():
    a = Arrow(2013, 5, 11, 13, 23, 58, 970460)
    floor, ceil = a.span("day")
    assert floor == Arrow(2013, 5, 11)
    assert ceil == Arrow(2013, 5, 11, 23, 59, 59, 999999)


def test_span_week_defaults_to_monday():
    a = Arrow(2013, 5, 11, 13)  # a Saturday
    floor, ceil = a.span("week")
    assert floor == Arrow(2013, 5, 6)
    assert ceil == Arrow(2013, 5, 12, 23, 59, 59, 999999)


def test_span_week_start_sunday():
    a = Arrow(2013, 5, 11, 13)
    floor, ceil = a.span("week", week_start=7)
    assert floor == Arrow(2013, 5, 5)
    assert ceil == Arrow(2013, 5, 11, 23, 59, 59, 999999)


def test_span_month():
    floor, ceil = Arrow(2013, 5, 11, 13).span("month")
    assert floor == Arrow(2013, 5, 1)
    assert ceil == Arrow(2013, 5, 31, 23, 59, 59, 999999)


def test_span_count_two_days():
    floor, ceil = Arrow(2013, 5, 11, 13).span("day", count=2)
    assert floor == Arrow(2013, 5, 11)
    assert ceil == Arrow(2013, 5, 12, 23, 59, 59, 999999)


def test_span_exact_starts_at_self():
    floor, ceil = Arrow(2013, 5, 11, 12, 30, 15).span("hour", exact=True)
    assert floor == Arrow(2013, 5, 11, 12, 30, 15)
    assert ceil == Arrow(2013, 5, 11, 13, 30, 14, 999999)


def test_span_inclusive_end_bounds():
    floor, ceil = Arrow(2013, 5, 11, 13).span("day", bounds="[]")
    assert floor == Arrow(2013, 5, 11)
    assert ceil == Arrow(2013, 5, 12)


def test_span_invalid_frame_raises():
    with pytest.raises(ValueError):
        Arrow(2013, 5, 11).span("fortnight")


def test_floor_and_ceil_hour():
    a = Arrow(2013, 5, 11, 13, 23, 58)
    assert a.floor("hour") == Arrow(2013, 5, 11, 13)
    assert a.ceil("hour") == Arrow(2013, 5, 11, 13, 59, 59, 999999)


def test_range_hours_is_inclusive_of_end():
    start = Arrow(2013, 5, 5, 12)
    end = Arrow(2013, 5, 5, 15)
    result = list(Arrow.range("hour", start, end))
    assert [r.hour for r in result] == [12, 13, 14, 15]
    assert all(isinstance(r, Arrow) for r in result)


def test_span_range_days():
    start = Arrow(2013, 5, 1)
    end = Arrow(2013, 5, 3)
    spans = list(Arrow.span_range("day", start, end))
    assert len(spans) == 3
    assert spans[0][0] == Arrow(2013, 5, 1)
    assert spans[0][1] == Arrow(2013, 5, 1, 23, 59, 59, 999999)


def test_interval_hours_groups_frames():
    start = Arrow(2013, 5, 5, 12)
    end = Arrow(2013, 5, 5, 16)
    intervals = list(Arrow.interval("hour", start, end, 2))
    assert intervals[0][0] == Arrow(2013, 5, 5, 12)
    assert intervals[0][1] == Arrow(2013, 5, 5, 13, 59, 59, 999999)
