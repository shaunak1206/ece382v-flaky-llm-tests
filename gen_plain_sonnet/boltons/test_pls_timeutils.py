from datetime import date, datetime, timedelta

import pytest
from freezegun import freeze_time

from boltons.timeutils import (
    UTC,
    ConstantTZInfo,
    Eastern,
    Pacific,
    daterange,
    decimal_relative_time,
    dt_to_timestamp,
    isoparse,
    parse_td,
    parse_timedelta,
    relative_time,
    strpdate,
)


def test_dt_to_timestamp():
    assert dt_to_timestamp(datetime(1970, 1, 2)) == 86400.0
    assert dt_to_timestamp(datetime(1970, 1, 1)) == 0.0
    plus_one = ConstantTZInfo('X', timedelta(hours=1))
    assert dt_to_timestamp(datetime(1970, 1, 1, 1, tzinfo=plus_one)) == 0.0
    assert dt_to_timestamp(datetime(1970, 1, 1, 0, 0, 5, tzinfo=UTC)) == 5.0


def test_isoparse_and_strpdate():
    assert isoparse('1970-01-01T00:00:00.851') == datetime(1970, 1, 1, 0, 0, 0, 851000)
    assert isoparse('2020-05-17T13:45:30') == datetime(2020, 5, 17, 13, 45, 30)
    assert isoparse('2020-05-17') == datetime(2020, 5, 17)
    stamp = datetime(2021, 3, 4, 5, 6, 7, 123456)
    assert isoparse(stamp.isoformat()) == stamp

    assert strpdate('2016-02-14', '%Y-%m-%d') == date(2016, 2, 14)
    assert strpdate('26/12 (2015)', '%d/%m (%Y)') == date(2015, 12, 26)
    assert strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S') == date(2015, 12, 31)
    with pytest.raises(ValueError):
        strpdate('not a date', '%Y-%m-%d')


def test_parse_timedelta():
    assert parse_td('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)
    assert parse_td('2 weeks 1 day') == timedelta(days=15)
    assert parse_td('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)
    assert parse_timedelta('90 minutes') == timedelta(minutes=90)
    assert parse_timedelta('45s') == timedelta(seconds=45)
    assert parse_timedelta('') == timedelta(0)
    assert parse_td is parse_timedelta
    with pytest.raises(ValueError):
        parse_timedelta('3 x')


def test_relative_time_with_explicit_other():
    now = datetime(2020, 1, 10, 12, 0, 0)
    assert decimal_relative_time(now - timedelta(days=1, seconds=3600), now) == (1.0, 'day')
    assert decimal_relative_time(now - timedelta(seconds=0.002), now, ndigits=5) == (
        0.002, 'seconds')
    assert decimal_relative_time(now, now - timedelta(days=900), ndigits=1) == (
        -2.5, 'years')
    assert decimal_relative_time(now - timedelta(days=2), now, cardinalize=False) == (
        2.0, 'day')

    assert relative_time(now, now) == '0 seconds ago'
    assert relative_time(now - timedelta(days=1, seconds=36000), now, ndigits=1) == '1.4 days ago'
    assert relative_time(now + timedelta(days=7), now, ndigits=1) == '1 week from now'
    assert relative_time(now - timedelta(hours=1), now) == '1 hour ago'
    assert relative_time(now - timedelta(minutes=5), now) == '5 minutes ago'


@freeze_time('2020-01-10 00:00:00')
def test_relative_time_defaults_to_now():
    assert relative_time(datetime(2020, 1, 9, 12, 0, 0)) == '12 hours ago'
    assert relative_time(datetime(2020, 1, 10, 0, 0, 30)) == '30 seconds from now'


def test_daterange_forward():
    christmas = date(2015, 12, 25)
    new_year = date(2016, 1, 1)
    days = list(daterange(christmas, new_year))
    assert len(days) == 7
    assert days[0] == christmas
    assert days[-1] == date(2015, 12, 31)
    assert list(daterange(christmas, date(2015, 12, 26))) == [christmas]

    assert list(daterange(date(2020, 1, 1), date(2020, 1, 10), step=3)) == [
        date(2020, 1, 1), date(2020, 1, 4), date(2020, 1, 7)]
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 7), step=timedelta(days=3),
                          inclusive=True)) == [
        date(2020, 1, 1), date(2020, 1, 4), date(2020, 1, 7)]
    assert list(daterange(date(2017, 5, 1), date(2017, 8, 1), step=(0, 1, 0),
                          inclusive=True)) == [
        date(2017, 5, 1), date(2017, 6, 1), date(2017, 7, 1), date(2017, 8, 1)]


def test_daterange_edge_cases():
    assert list(daterange(date(2020, 1, 5), date(2020, 1, 1), step=-1)) == [
        date(2020, 1, 5), date(2020, 1, 4), date(2020, 1, 3), date(2020, 1, 2)]
    # a step pointing away from stop yields nothing
    assert list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=-1)) == []
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=0))
    with pytest.raises(ValueError):
        list(daterange(date(2020, 1, 1), date(2020, 1, 5), step=1.5))
    with pytest.raises(TypeError):
        list(daterange('2020-01-01', date(2020, 1, 5)))
    with pytest.raises(TypeError):
        list(daterange(date(2020, 1, 1), '2020-01-05'))

    infinite = daterange(date(2020, 1, 1), None)
    first_three = [next(infinite) for _ in range(3)]
    assert first_three == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]


def test_timezone_classes():
    summer = datetime(2020, 7, 1, 12, 0, tzinfo=Eastern)
    winter = datetime(2020, 1, 15, 12, 0, tzinfo=Eastern)
    assert summer.utcoffset() == timedelta(hours=-4)
    assert winter.utcoffset() == timedelta(hours=-5)
    assert summer.tzname() == 'EDT'
    assert winter.tzname() == 'EST'
    assert repr(Eastern) == 'Eastern'

    converted = datetime(2020, 7, 1, 16, 0, tzinfo=UTC).astimezone(Pacific)
    assert converted.hour == 9
    assert converted.tzname() == 'PDT'

    assert UTC.utcoffset(None) == timedelta(0)
    assert UTC.tzname(None) == 'UTC'
    assert UTC.dst(None) == timedelta(0)
    india = ConstantTZInfo('IST', timedelta(hours=5.5))
    assert india.utcoffset_hours == 5.5
    assert 'IST' in repr(india)
