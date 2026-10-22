import datetime

import pytest
from freezegun import freeze_time

from schedule import (
    IntervalError,
    Job,
    ScheduleError,
    ScheduleValueError,
    Scheduler,
)


def noop():
    return None


WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def test_exception_hierarchy():
    assert issubclass(ScheduleValueError, ScheduleError)
    assert issubclass(IntervalError, ScheduleValueError)
    assert issubclass(ScheduleError, Exception)


@pytest.mark.parametrize(
    "attr,unit",
    [
        ("seconds", "seconds"),
        ("minutes", "minutes"),
        ("hours", "hours"),
        ("days", "days"),
        ("weeks", "weeks"),
    ],
)
def test_plural_unit_properties_set_unit(attr, unit):
    job = Scheduler().every(3)
    result = getattr(job, attr)
    assert result is job
    assert job.unit == unit


@pytest.mark.parametrize(
    "attr,unit",
    [
        ("second", "seconds"),
        ("minute", "minutes"),
        ("hour", "hours"),
        ("day", "days"),
        ("week", "weeks"),
    ],
)
def test_singular_unit_properties_with_interval_one(attr, unit):
    job = Scheduler().every()
    assert job.interval == 1
    result = getattr(job, attr)
    assert result is job
    assert job.unit == unit


@pytest.mark.parametrize(
    "attr", ["second", "minute", "hour", "day", "week"] + WEEKDAYS
)
def test_singular_properties_reject_interval_other_than_one(attr):
    job = Scheduler().every(2)
    with pytest.raises(IntervalError):
        getattr(job, attr)


@pytest.mark.parametrize("day", WEEKDAYS)
def test_weekday_property_sets_start_day_and_weeks(day):
    job = Scheduler().every()
    result = getattr(job, day)
    assert result is job
    assert job.start_day == day
    assert job.unit == "weeks"


def test_tag_adds_tags_discards_duplicates_and_returns_job():
    job = Scheduler().every().hour
    assert job.tag("a", "b", "a") is job
    job.tag("b")
    assert job.tags == {"a", "b"}


def test_tag_rejects_unhashable():
    job = Scheduler().every().hour
    with pytest.raises(TypeError):
        job.tag(["not", "hashable"])


def test_to_sets_latest_and_returns_job():
    job = Scheduler().every(5)
    assert job.to(10) is job
    assert job.latest == 10


def test_do_without_unit_raises_value_error_and_does_not_register():
    s = Scheduler()
    with pytest.raises(ScheduleValueError):
        s.every().do(noop)
    assert s.jobs == []


def test_do_without_scheduler_raises_schedule_error():
    job = Job(1)
    job.seconds
    with pytest.raises(ScheduleError):
        job.do(noop)


def test_do_registers_job_and_keeps_function_name():
    s = Scheduler()
    job = s.every().second.do(noop)
    assert isinstance(job, Job)
    assert s.jobs == [job]
    assert job.job_func.__name__ == "noop"


def test_do_passes_args_and_kwargs_to_function():
    s = Scheduler()
    results = []

    def add(a, b=0):
        results.append(a + b)
        return a + b

    job = s.every().second.do(add, 1, b=2)
    assert job.run() == 3
    assert results == [3]


def test_at_requires_string_argument():
    with pytest.raises(TypeError):
        Scheduler().every().day.at(1030)


@pytest.mark.parametrize(
    "bad", ["9:30", "10:60", "10", "10:30:60", "abc", "10-30", "24:00"]
)
def test_at_daily_rejects_invalid_values(bad):
    with pytest.raises(ScheduleValueError):
        Scheduler().every().day.at(bad)


def test_at_daily_parses_time_with_and_without_seconds():
    assert Scheduler().every().day.at("10:30").at_time == datetime.time(10, 30, 0)
    assert Scheduler().every().day.at("10:30:15").at_time == datetime.time(
        10, 30, 15
    )


def test_at_hourly_formats():
    assert Scheduler().every().hour.at(":15").at_time == datetime.time(0, 15, 0)
    assert Scheduler().every().hour.at("05:30").at_time == datetime.time(0, 5, 30)


@pytest.mark.parametrize("bad", ["1:30", "10:30:00", "60:00"])
def test_at_hourly_rejects_malformed_strings(bad):
    with pytest.raises(ScheduleValueError):
        Scheduler().every().hour.at(bad)


def test_at_minutely_means_seconds_and_rejects_other_formats():
    assert Scheduler().every().minute.at(":30").at_time == datetime.time(0, 0, 30)
    with pytest.raises(ScheduleValueError):
        Scheduler().every().minute.at("10:30")


def test_at_rejects_unsupported_units():
    with pytest.raises(ScheduleValueError):
        Scheduler().every().second.at(":30")
    with pytest.raises(ScheduleValueError):
        Scheduler().every().week.at("10:30")


def test_at_works_with_weekday_jobs():
    job = Scheduler().every().monday.at("09:00")
    assert job.at_time == datetime.time(9, 0, 0)
    assert job.start_day == "monday"


def test_until_rejects_unsupported_type_and_bad_string():
    with pytest.raises(TypeError):
        Scheduler().every().day.until(5)
    with pytest.raises(ScheduleValueError):
        Scheduler().every().day.until("not a time")


@freeze_time("2024-01-15 12:00:00")
def test_until_datetime_timedelta_and_time():
    target = datetime.datetime(2024, 1, 16, 8, 0, 0)
    assert Scheduler().every().day.until(target).cancel_after == target
    job = Scheduler().every().day.until(datetime.timedelta(hours=1))
    assert job.cancel_after == datetime.datetime(2024, 1, 15, 13, 0, 0)
    job = Scheduler().every().day.until(datetime.time(18, 0))
    assert job.cancel_after == datetime.datetime(2024, 1, 15, 18, 0, 0)


@freeze_time("2024-01-15 12:00:00")
@pytest.mark.parametrize(
    "text,expected",
    [
        ("2024-01-16 10:30:45", datetime.datetime(2024, 1, 16, 10, 30, 45)),
        ("2024-01-16 10:30", datetime.datetime(2024, 1, 16, 10, 30)),
        ("2024-01-16", datetime.datetime(2024, 1, 16)),
        ("18:30:15", datetime.datetime(2024, 1, 15, 18, 30, 15)),
        ("18:30", datetime.datetime(2024, 1, 15, 18, 30)),
    ],
)
def test_until_string_formats(text, expected):
    job = Scheduler().every().day.until(text)
    assert job.cancel_after == expected


@freeze_time("2024-01-15 12:00:00")
def test_until_rejects_past_moments():
    with pytest.raises(ScheduleValueError):
        Scheduler().every().day.until(datetime.datetime(2024, 1, 15, 11, 0, 0))
    with pytest.raises(ScheduleValueError):
        Scheduler().every().day.until("11:00")


@freeze_time("2024-01-15 12:00:00")
def test_str_of_configured_job():
    job = Scheduler().every().hour.do(noop)
    assert str(job) == "Job(interval=1, unit=hours, do=noop, args=(), kwargs={})"


@freeze_time("2024-01-15 12:00:00")
def test_repr_singular_unit_before_first_run():
    job = Scheduler().every().hour.do(noop)
    assert repr(job) == (
        "Every 1 hour do noop() (last run: [never], next run: 2024-01-15 13:00:00)"
    )


@freeze_time("2024-01-15 12:00:00")
def test_repr_with_at_time():
    job = Scheduler().every().day.at("10:30").do(noop)
    assert repr(job) == (
        "Every 1 day at 10:30:00 do noop() "
        "(last run: [never], next run: 2024-01-16 10:30:00)"
    )
