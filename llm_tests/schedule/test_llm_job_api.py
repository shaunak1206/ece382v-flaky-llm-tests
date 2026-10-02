"""
Tests for the Job builder API: every(), the time-unit properties
(seconds/minutes/hours/days/weeks and their singular forms), weekday
chaining, .at(), .to(), .until(), .tag() and .do().
"""
import datetime

import pytest

import schedule
from schedule import IntervalError, Job, ScheduleValueError


def noop():
    pass


# ---------------------------------------------------------------------------
# every() / interval / time-unit properties
# ---------------------------------------------------------------------------

def test_every_default_interval_is_one(scheduler):
    assert scheduler.every().interval == 1


def test_every_with_custom_interval(scheduler):
    assert scheduler.every(5).interval == 5


@pytest.mark.parametrize(
    "attr,expected_unit",
    [
        ("seconds", "seconds"),
        ("minutes", "minutes"),
        ("hours", "hours"),
        ("days", "days"),
        ("weeks", "weeks"),
    ],
)
def test_plural_unit_properties_set_unit_and_return_job(scheduler, attr, expected_unit):
    job = scheduler.every(3)
    result = getattr(job, attr)
    assert result is job
    assert job.unit == expected_unit


@pytest.mark.parametrize(
    "singular,plural_unit",
    [
        ("second", "seconds"),
        ("minute", "minutes"),
        ("hour", "hours"),
        ("day", "days"),
        ("week", "weeks"),
    ],
)
def test_singular_unit_property_works_for_interval_one(scheduler, singular, plural_unit):
    job = scheduler.every(1)
    result = getattr(job, singular)
    assert job.unit == plural_unit
    assert result is job


@pytest.mark.parametrize(
    "weekday",
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)
def test_weekday_property_sets_start_day_and_unit(scheduler, weekday):
    job = getattr(scheduler.every(), weekday)
    assert job.start_day == weekday
    assert job.unit == "weeks"


# ---------------------------------------------------------------------------
# .at()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "time_str,expected",
    [
        ("10:30", datetime.time(10, 30, 0)),
        ("10:30:15", datetime.time(10, 30, 15)),
    ],
)
def test_at_daily_formats(scheduler, time_str, expected):
    assert scheduler.every().day.at(time_str).at_time == expected


@pytest.mark.parametrize("bad_time", ["25:00", "10-30", "abc", "10:30:99"])
def test_at_daily_invalid_format_raises(scheduler, bad_time):
    with pytest.raises(ScheduleValueError):
        scheduler.every().day.at(bad_time)


@pytest.mark.parametrize(
    "time_str,expected",
    [
        ("30:15", datetime.time(0, 30, 15)),
        # The short ":MM" form sets the minute, not the second (see
        # Job.at()'s docstring: "For hourly jobs -> MM:SS or :MM").
        (":45", datetime.time(0, 45, 0)),
    ],
)
def test_at_hourly_formats(scheduler, time_str, expected):
    assert scheduler.every().hour.at(time_str).at_time == expected


def test_at_hourly_invalid_format_raises(scheduler):
    with pytest.raises(ScheduleValueError):
        scheduler.every().hour.at("10:30:15")


def test_at_minutes_colon_ss(scheduler):
    assert scheduler.every().minutes.at(":30").at_time == datetime.time(0, 0, 30)


def test_at_minutes_invalid_format_raises(scheduler):
    with pytest.raises(ScheduleValueError):
        scheduler.every().minutes.at("10:30")


def test_at_non_string_raises_typeerror(scheduler):
    with pytest.raises(TypeError):
        scheduler.every().day.at(1030)


def test_at_on_weekday_job_uses_daily_format(scheduler):
    assert scheduler.every().monday.at("10:30").at_time == datetime.time(10, 30, 0)


# ---------------------------------------------------------------------------
# .to()
# ---------------------------------------------------------------------------

def test_to_sets_latest_and_returns_job_for_chaining(scheduler):
    job = scheduler.every(5)
    assert job.to(10) is job
    assert job.latest == 10


# ---------------------------------------------------------------------------
# .tag()
# ---------------------------------------------------------------------------

def test_tag_multiple_and_duplicates_discarded(scheduler):
    job = scheduler.every().seconds.do(noop)
    job.tag("a", "a", "b")
    assert job.tags == {"a", "b"}


def test_tag_unhashable_raises_typeerror(scheduler):
    job = scheduler.every().seconds.do(noop)
    with pytest.raises(TypeError):
        job.tag(["not", "hashable"])


def test_tag_returns_job_and_accepts_mixed_hashables(scheduler):
    job = scheduler.every().seconds.do(noop)
    assert job.tag(1, "two", 3.0) is job
    assert {1, "two", 3.0} == job.tags


# ---------------------------------------------------------------------------
# .do()
# ---------------------------------------------------------------------------

def test_do_sets_job_func_and_registers_with_scheduler(scheduler):
    job = scheduler.every().seconds.do(noop)
    assert job.job_func.func is noop
    assert job in scheduler.jobs
    assert job.next_run is not None


def test_do_with_args_and_kwargs(scheduler):
    calls = []

    def record(a, b, c=None):
        calls.append((a, b, c))

    scheduler.every().seconds.do(record, 1, 2, c=3)
    scheduler.run_all()
    assert calls == [(1, 2, 3)]


def test_do_without_scheduler_raises_schedule_error():
    job = Job(1, scheduler=None)
    job.unit = "seconds"
    with pytest.raises(schedule.ScheduleError):
        job.do(noop)


# ---------------------------------------------------------------------------
# .until()
# ---------------------------------------------------------------------------

def test_until_with_datetime(scheduler):
    future = datetime.datetime.now() + datetime.timedelta(days=1)
    assert scheduler.every().seconds.until(future).cancel_after == future


def test_until_with_timedelta(scheduler):
    before = datetime.datetime.now()
    job = scheduler.every().seconds.until(datetime.timedelta(hours=1))
    after = datetime.datetime.now()
    assert before + datetime.timedelta(hours=1) <= job.cancel_after
    assert job.cancel_after <= after + datetime.timedelta(hours=1)


def test_until_with_time_object(scheduler):
    now = datetime.datetime.now()
    future_dt = now + datetime.timedelta(hours=1)
    if future_dt.date() != now.date():
        pytest.skip("test would cross midnight; until(time) is same-day only")
    future_time = future_dt.time()
    job = scheduler.every().seconds.until(future_time)
    assert job.cancel_after.time().hour == future_time.hour
    assert job.cancel_after.time().minute == future_time.minute


@pytest.mark.parametrize(
    "fmt",
    ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"],
)
def test_until_with_valid_date_strings(scheduler, fmt):
    future = datetime.datetime.now() + datetime.timedelta(days=2)
    until_str = future.strftime(fmt)
    assert scheduler.every().seconds.until(until_str).cancel_after is not None


# ---------------------------------------------------------------------------
# __str__ / __repr__
# ---------------------------------------------------------------------------

def test_job_str_contains_unit_and_interval(scheduler):
    text = str(scheduler.every(3).days.do(noop))
    assert "3" in text
    assert "days" in text


def test_job_repr_contains_function_name_and_next_run(scheduler):
    job = scheduler.every().day.at("10:30").do(noop)
    text = repr(job)
    assert "noop" in text
    assert "last run: [never]" in text
    assert "next run: [never]" not in text


def test_job_repr_unconfigured_job_shows_never(scheduler):
    assert "[never]" in repr(scheduler.every().days)
