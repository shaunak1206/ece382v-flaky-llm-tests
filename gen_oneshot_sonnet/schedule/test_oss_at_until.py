"""Tests for Job.at() and Job.until()."""

import datetime

import pytest
from freezegun import freeze_time

from schedule import Scheduler, ScheduleValueError, CancelJob


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


@pytest.fixture
def sched():
    return Scheduler()


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1


# ---------------------------------------------------------------- at()


def test_at_daily_time_in_future_runs_today(frozen, sched):
    job = sched.every().day.at("13:45").do(lambda: None)
    assert job.at_time == datetime.time(13, 45, 0)
    assert job.next_run == datetime.datetime(2024, 1, 1, 13, 45, 0)


def test_at_daily_time_in_past_runs_tomorrow(frozen, sched):
    job = sched.every().day.at("10:30:15").do(lambda: None)
    assert job.at_time == datetime.time(10, 30, 15)
    assert job.next_run == datetime.datetime(2024, 1, 2, 10, 30, 15)


def test_at_hourly_minute_only_format(frozen, sched):
    job = sched.every().hour.at(":30").do(lambda: None)
    assert job.at_time == datetime.time(0, 30, 0)
    assert job.next_run == datetime.datetime(2024, 1, 1, 12, 30, 0)


def test_at_hourly_with_minutes_and_seconds(frozen, sched):
    job = sched.every().hour.at("15:30").do(lambda: None)
    assert job.at_time == datetime.time(0, 15, 30)
    assert job.next_run == datetime.datetime(2024, 1, 1, 12, 15, 30)


def test_at_minutely_seconds_format(frozen, sched):
    job = sched.every().minute.at(":15").do(lambda: None)
    assert job.at_time == datetime.time(0, 0, 15)
    assert job.next_run == datetime.datetime(2024, 1, 1, 12, 0, 15)


def test_at_returns_job(sched):
    job = sched.every().day
    assert job.at("10:00") is job


@pytest.mark.parametrize(
    "bad", ["1:30", "10:60", "10-30", "abc", "", "30:00:00:00", "10:30:75", "25:00"]
)
def test_at_invalid_daily_format_raises(sched, bad):
    with pytest.raises(ScheduleValueError):
        sched.every().day.at(bad)


@pytest.mark.parametrize("bad", ["10:30:15", "1:30", "abc", "::30"])
def test_at_invalid_hourly_format_raises(sched, bad):
    with pytest.raises(ScheduleValueError):
        sched.every().hour.at(bad)


@pytest.mark.parametrize("bad", ["10:30", ":5", "30", "::30"])
def test_at_invalid_minutely_format_raises(sched, bad):
    with pytest.raises(ScheduleValueError):
        sched.every().minute.at(bad)


@pytest.mark.parametrize("unit", ["seconds", "weeks"])
def test_at_on_unsupported_unit_raises(sched, unit):
    job = getattr(sched.every(), unit)
    with pytest.raises(ScheduleValueError):
        job.at("10:00")


def test_at_requires_string(sched):
    with pytest.raises(TypeError):
        sched.every().day.at(1030)


def test_at_with_timezone_sets_pytz_zone():
    pytz = pytest.importorskip("pytz")
    s = Scheduler()
    job = s.every().day.at("10:00", "UTC")
    assert isinstance(job.at_time_zone, pytz.BaseTzInfo)


def test_daily_at_job_runs_when_clock_reaches_time(frozen, sched):
    counter = Counter()
    sched.every().day.at("13:00").do(counter)
    sched.run_pending()
    assert counter.count == 0
    frozen.move_to("2024-01-01 12:59:59")
    sched.run_pending()
    assert counter.count == 0
    frozen.move_to("2024-01-01 13:00:00")
    sched.run_pending()
    assert counter.count == 1
    sched.run_pending()
    assert counter.count == 1
    frozen.move_to("2024-01-02 13:00:00")
    sched.run_pending()
    assert counter.count == 2


# -------------------------------------------------------------- until()


def test_until_datetime_sets_cancel_after_and_returns_job(frozen, sched):
    deadline = datetime.datetime(2024, 1, 1, 13, 0, 0)
    job = sched.every().minute
    assert job.until(deadline) is job
    assert job.cancel_after == deadline


def test_until_timedelta_is_relative_to_now(frozen, sched):
    job = sched.every().minute.until(datetime.timedelta(hours=2))
    assert job.cancel_after == datetime.datetime(2024, 1, 1, 14, 0, 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-01-02 08:15:30", datetime.datetime(2024, 1, 2, 8, 15, 30)),
        ("2024-01-02 08:15", datetime.datetime(2024, 1, 2, 8, 15, 0)),
        ("2024-01-02", datetime.datetime(2024, 1, 2, 0, 0, 0)),
        ("14:20:10", datetime.datetime(2024, 1, 1, 14, 20, 10)),
        ("14:20", datetime.datetime(2024, 1, 1, 14, 20, 0)),
    ],
)
def test_until_string_formats(frozen, sched, text, expected):
    job = sched.every().minute.until(text)
    assert job.cancel_after == expected


def test_until_invalid_string_raises(sched):
    with pytest.raises(ScheduleValueError):
        sched.every().minute.until("not a time")


def test_until_in_the_past_raises(frozen, sched):
    with pytest.raises(ScheduleValueError):
        sched.every().minute.until(datetime.datetime(2023, 12, 31, 23, 59, 59))
    with pytest.raises(ScheduleValueError):
        sched.every().minute.until("11:00")


@pytest.mark.parametrize("bad", [123, 4.5, None, datetime.date(2030, 1, 1), ["13:00"]])
def test_until_wrong_type_raises_type_error(sched, bad):
    with pytest.raises(TypeError):
        sched.every().minute.until(bad)


def test_job_runs_until_deadline_then_cancels_itself(frozen, sched):
    counter = Counter()
    sched.every(10).minutes.until("12:25").do(counter)

    frozen.move_to("2024-01-01 12:10:00")
    sched.run_pending()
    assert counter.count == 1
    assert len(sched.jobs) == 1

    # The run at 12:20 schedules 12:30 next, which is after the deadline.
    frozen.move_to("2024-01-01 12:20:00")
    sched.run_pending()
    assert counter.count == 2
    assert sched.jobs == []

    frozen.move_to("2024-01-01 12:30:00")
    sched.run_pending()
    assert counter.count == 2


def test_job_whose_deadline_passed_before_running_is_cancelled_without_running(frozen, sched):
    counter = Counter()
    sched.every(10).minutes.until("12:15").do(counter)

    frozen.move_to("2024-01-01 12:30:00")
    sched.run_pending()
    assert counter.count == 0
    assert sched.jobs == []


def test_job_run_returns_cancel_job_when_overdue(frozen, sched):
    counter = Counter()
    job = sched.every(10).minutes.until("12:15").do(counter)
    frozen.move_to("2024-01-01 12:30:00")
    assert job.run() is CancelJob
    assert counter.count == 0
    assert job.last_run is None
