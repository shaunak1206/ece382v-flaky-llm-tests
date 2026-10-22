import datetime

import pytest
from freezegun import freeze_time

from schedule import ScheduleError, Scheduler

# 2024-01-15 is a Monday.
NOW = "2024-01-15 12:00:00"


def dt(*args):
    return datetime.datetime(*args)


def noop():
    return None


@freeze_time(NOW)
def test_every_ten_seconds():
    job = Scheduler().every(10).seconds.do(noop)
    assert job.next_run == dt(2024, 1, 15, 12, 0, 10)


@freeze_time(NOW)
def test_every_five_minutes():
    job = Scheduler().every(5).minutes.do(noop)
    assert job.next_run == dt(2024, 1, 15, 12, 5, 0)


@freeze_time(NOW)
def test_every_two_hours():
    job = Scheduler().every(2).hours.do(noop)
    assert job.next_run == dt(2024, 1, 15, 14, 0, 0)


@freeze_time(NOW)
def test_every_three_days():
    job = Scheduler().every(3).days.do(noop)
    assert job.next_run == dt(2024, 1, 18, 12, 0, 0)


@freeze_time(NOW)
def test_every_two_weeks():
    job = Scheduler().every(2).weeks.do(noop)
    assert job.next_run == dt(2024, 1, 29, 12, 0, 0)


@freeze_time("2024-02-28 12:00:00")
def test_every_two_days_crosses_leap_day_into_march():
    job = Scheduler().every(2).days.do(noop)
    assert job.next_run == dt(2024, 3, 1, 12, 0, 0)


@freeze_time("2024-12-31 23:59:30")
def test_every_minute_crosses_year_boundary():
    job = Scheduler().every().minute.do(noop)
    assert job.next_run == dt(2025, 1, 1, 0, 0, 30)


@freeze_time(NOW)
def test_next_run_is_naive_datetime():
    job = Scheduler().every().hour.do(noop)
    assert job.next_run.tzinfo is None


@freeze_time(NOW)
def test_daily_at_later_today():
    job = Scheduler().every().day.at("13:30").do(noop)
    assert job.next_run == dt(2024, 1, 15, 13, 30, 0)


@freeze_time(NOW)
def test_daily_at_already_passed_goes_to_tomorrow():
    job = Scheduler().every().day.at("10:30").do(noop)
    assert job.next_run == dt(2024, 1, 16, 10, 30, 0)


@freeze_time(NOW)
def test_daily_at_exactly_now_goes_to_tomorrow():
    job = Scheduler().every().day.at("12:00").do(noop)
    assert job.next_run == dt(2024, 1, 16, 12, 0, 0)


@freeze_time(NOW)
def test_every_two_days_at_time():
    job = Scheduler().every(2).days.at("10:30").do(noop)
    assert job.next_run == dt(2024, 1, 17, 10, 30, 0)


@freeze_time(NOW)
def test_hourly_at_minute_later_this_hour():
    job = Scheduler().every().hour.at(":30").do(noop)
    assert job.next_run == dt(2024, 1, 15, 12, 30, 0)


@freeze_time("2024-01-15 12:45:00")
def test_hourly_at_minute_already_passed_goes_to_next_hour():
    job = Scheduler().every().hour.at(":30").do(noop)
    assert job.next_run == dt(2024, 1, 15, 13, 30, 0)


@freeze_time(NOW)
def test_minutely_at_seconds_later_this_minute():
    job = Scheduler().every().minute.at(":30").do(noop)
    assert job.next_run == dt(2024, 1, 15, 12, 0, 30)


@freeze_time("2024-01-15 12:00:45")
def test_minutely_at_seconds_already_passed_goes_to_next_minute():
    job = Scheduler().every().minute.at(":30").do(noop)
    assert job.next_run == dt(2024, 1, 15, 12, 1, 30)


@freeze_time(NOW)
def test_weekday_same_day_without_time_goes_to_next_week():
    job = Scheduler().every().monday.do(noop)
    assert job.next_run == dt(2024, 1, 22, 12, 0, 0)


@freeze_time(NOW)
def test_weekday_later_this_week():
    job = Scheduler().every().wednesday.do(noop)
    assert job.next_run == dt(2024, 1, 17, 12, 0, 0)


@freeze_time(NOW)
def test_weekday_with_time_later_this_week():
    job = Scheduler().every().friday.at("09:00").do(noop)
    assert job.next_run == dt(2024, 1, 19, 9, 0, 0)


@freeze_time(NOW)
def test_weekday_today_time_still_ahead_runs_today():
    job = Scheduler().every().monday.at("15:00").do(noop)
    assert job.next_run == dt(2024, 1, 15, 15, 0, 0)


@freeze_time(NOW)
def test_weekday_today_time_passed_runs_next_week():
    job = Scheduler().every().monday.at("09:00").do(noop)
    assert job.next_run == dt(2024, 1, 22, 9, 0, 0)


@freeze_time(NOW)
def test_randomized_interval_within_bounds():
    for _ in range(20):
        job = Scheduler().every(5).to(10).seconds.do(noop)
        assert dt(2024, 1, 15, 12, 0, 5) <= job.next_run <= dt(2024, 1, 15, 12, 0, 10)


@freeze_time(NOW)
def test_latest_smaller_than_interval_raises():
    with pytest.raises(ScheduleError):
        Scheduler().every(10).to(5).seconds.do(noop)


def test_should_run_transitions_at_next_run():
    with freeze_time(NOW) as frozen:
        job = Scheduler().every(10).seconds.do(noop)
        assert job.should_run is False
        frozen.move_to("2024-01-15 12:00:09")
        assert job.should_run is False
        frozen.move_to("2024-01-15 12:00:10")
        assert job.should_run is True


def test_run_updates_last_run_and_reschedules():
    with freeze_time(NOW) as frozen:
        job = Scheduler().every(10).seconds.do(noop)
        assert job.last_run is None
        frozen.move_to("2024-01-15 12:00:10")
        job.run()
        assert job.last_run == dt(2024, 1, 15, 12, 0, 10)
        assert job.next_run == dt(2024, 1, 15, 12, 0, 20)


def test_missed_runs_are_not_replayed():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every().minute.do(lambda: calls.append(1))
        frozen.move_to("2024-01-15 13:00:00")
        s.run_pending()
        s.run_pending()
    assert calls == [1]
