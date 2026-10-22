import itertools
from datetime import date, datetime, timedelta, timezone

import pytest

from boltons.timeutils import (
    ConstantTZInfo,
    Eastern,
    Pacific,
    daterange,
    dt_to_timestamp,
    isoparse,
    parse_timedelta,
    relative_time,
    strpdate,
)


def test_parse_timedelta_documented_examples():
    assert parse_timedelta('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)
    assert parse_timedelta('2 weeks 1 day') == timedelta(days=15)
    assert parse_timedelta('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)
    assert parse_timedelta('90s') == timedelta(seconds=90)


def test_parse_timedelta_rejects_text_without_units():
    with pytest.raises(ValueError):
        parse_timedelta('hello')
    with pytest.raises(ValueError):
        parse_timedelta('')


def test_isoparse_and_dt_to_timestamp():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)
    stamp = datetime(2020, 5, 17, 13, 45, 30, 123456)
    assert isoparse(stamp.isoformat()) == stamp
    assert dt_to_timestamp(datetime(1970, 1, 2)) == 86400.0
    assert dt_to_timestamp(datetime(1970, 1, 1, 1, tzinfo=timezone.utc)) == 3600.0
    aware = datetime(1970, 1, 1, 5, tzinfo=ConstantTZInfo('X', timedelta(hours=5)))
    assert dt_to_timestamp(aware) == 0.0


def test_relative_time_past_and_future():
    other = datetime(2020, 6, 15, 12, 0, 0)
    assert relative_time(other, other) == '0 seconds ago'
    assert relative_time(other - timedelta(hours=2), other) == '2 hours ago'
    assert relative_time(other - timedelta(days=3), other) == '3 days ago'
    assert relative_time(other + timedelta(hours=3), other) == '3 hours from now'


def test_strpdate_returns_date_objects():
    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)
    result = strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S')
    assert result == date(2015, 12, 31)
    assert type(result) is date


def test_daterange_exclusive_inclusive_step_and_infinite():
    start = date(2020, 1, 1)
    assert list(daterange(start, date(2020, 1, 4))) == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    assert list(daterange(start, date(2020, 1, 4), inclusive=True))[-1] == date(2020, 1, 4)
    assert list(daterange(start, date(2020, 1, 6), step=2)) == [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 5)]
    first_three = list(itertools.islice(daterange(start, None), 3))
    assert first_three == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_constant_and_us_timezones():
    tz = ConstantTZInfo('X', timedelta(hours=5))
    assert tz.utcoffset(None) == timedelta(hours=5)
    assert tz.tzname(None) == 'X'
    assert datetime(2020, 1, 1, tzinfo=tz).utcoffset() == timedelta(hours=5)

    assert datetime(2020, 1, 15, 12, tzinfo=Eastern).utcoffset() == timedelta(hours=-5)
    assert datetime(2020, 7, 15, 12, tzinfo=Eastern).utcoffset() == timedelta(hours=-4)
    assert datetime(2020, 1, 15, 12, tzinfo=Pacific).utcoffset() == timedelta(hours=-8)
    assert datetime(2020, 7, 15, 12, tzinfo=Pacific).utcoffset() == timedelta(hours=-7)
