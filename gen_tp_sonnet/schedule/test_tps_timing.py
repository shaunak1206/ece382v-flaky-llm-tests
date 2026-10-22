"""Tests for how schedule computes the next run for different units and times."""
import datetime

import pytest
from freezegun import freeze_time

import schedule

# 2020-01-01 is a Wednesday
START = "2020-01-01 09:00:00"
T0 = datetime.datetime(2020, 1, 1, 9, 0, 0)


def noop():
    return None


@pytest.fixture
def sched():
    return schedule.Scheduler()


@freeze_time(START)
def test_every_seconds_next_run(sched):
    job = sched.every(10).seconds.do(noop)
    assert job.next_run == T0 + datetime.timedelta(seconds=10)


@freeze_time(START)
def test_every_minutes_next_run(sched):
    job = sched.every(5).minutes.do(noop)
    assert job.next_run == T0 + datetime.timedelta(minutes=5)


@freeze_time(START)
def test_every_hours_next_run(sched):
    job = sched.every(3).hours.do(noop)
    assert job.next_run == T0 + datetime.timedelta(hours=3)


@freeze_time(START)
def test_every_days_next_run(sched):
    job = sched.every(2).days.do(noop)
    assert job.next_run == T0 + datetime.timedelta(days=2)


@freeze_time(START)
def test_every_weeks_next_run(sched):
    job = sched.every(2).weeks.do(noop)
    assert job.next_run == T0 + datetime.timedelta(weeks=2)


@freeze_time(START)
def test_every_default_interval_is_one(sched):
    job = sched.every().hour.do(noop)
    assert job.next_run == T0 + datetime.timedelta(hours=1)


@freeze_time(START)
@pytest.mark.parametrize(
    "unit,delta",
    [
        ("second", datetime.timedelta(seconds=1)),
        ("minute", datetime.timedelta(minutes=1)),
        ("hour", datetime.timedelta(hours=1)),
        ("day", datetime.timedelta(days=1)),
        ("week", datetime.timedelta(weeks=1)),
    ],
)
def test_singular_units_with_default_interval(sched, unit, delta):
    job = getattr(sched.every(), unit).do(noop)
    assert job.next_run == T0 + delta


@pytest.mark.parametrize("unit", ["second", "minute", "hour", "day", "week"])
def test_singular_units_reject_interval_above_one(sched, unit):
    with pytest.raises(schedule.IntervalError):
        getattr(sched.every(2), unit)


@pytest.mark.parametrize(
    "day",
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)
def test_weekday_rejects_interval_above_one(sched, day):
    with pytest.raises(schedule.IntervalError):
        getattr(sched.every(2), day)


@freeze_time(START)
@pytest.mark.parametrize(
    "day,index",
    [
        ("monday", 0),
        ("tuesday", 1),
        ("wednesday", 2),
        ("thursday", 3),
        ("friday", 4),
        ("saturday", 5),
        ("sunday", 6),
    ],
)
def test_weekday_next_run_falls_on_that_weekday(sched, day, index):
    job = getattr(sched.every(), day).do(noop)
    assert job.next_run.weekday() == index


@freeze_time(START)
@pytest.mark.parametrize(
    "day", ["monday", "tuesday", "thursday", "friday", "saturday", "sunday"]
)
@freeze_time(START)
def test_weekday_with_at_time(sched):
    job = sched.every().monday.at("10:30").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 6, 10, 30, 0)


@freeze_time(START)
def test_daily_at_later_time_runs_today(sched):
    job = sched.every().day.at("10:30").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 1, 10, 30, 0)


@freeze_time("2020-01-01 11:00:00")
def test_daily_at_earlier_time_runs_tomorrow(sched):
    job = sched.every().day.at("10:30").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 2, 10, 30, 0)


@freeze_time(START)
def test_daily_at_with_seconds(sched):
    job = sched.every().day.at("10:30:45").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 1, 10, 30, 45)


@freeze_time(START)
def test_hourly_at_minute_only(sched):
    job = sched.every().hour.at(":15").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 1, 9, 15, 0)


@freeze_time(START)
def test_hourly_at_minute_and_second(sched):
    job = sched.every().hour.at("30:15").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 1, 9, 30, 15)


@freeze_time(START)
def test_minutely_at_seconds(sched):
    job = sched.every().minute.at(":17").do(noop)
    assert job.next_run == datetime.datetime(2020, 1, 1, 9, 0, 17)


@pytest.mark.parametrize(
    "bad", ["25:00", "10:61", "abc", "10", ":30", "10:30:61", "", "1030"]
)
def test_daily_at_invalid_format_raises(sched, bad):
    with pytest.raises(schedule.ScheduleValueError):
        sched.every().day.at(bad)


def test_hourly_at_rejects_daily_format(sched):
    with pytest.raises(schedule.ScheduleValueError):
        sched.every().hour.at("10:30:00")


def test_minutely_at_rejects_minute_second_format(sched):
    with pytest.raises(schedule.ScheduleValueError):
        sched.every().minute.at("10:30")


def test_at_not_allowed_for_seconds_unit(sched):
    with pytest.raises(schedule.ScheduleValueError):
        sched.every(5).seconds.at(":10").do(noop)


def test_at_rejects_non_string(sched):
    with pytest.raises((TypeError, schedule.ScheduleValueError)):
        sched.every().day.at(1030)
