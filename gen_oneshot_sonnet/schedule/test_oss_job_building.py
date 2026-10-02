"""Tests for building jobs with the fluent API: units, weekdays, to(), tag(), do()."""

import datetime

import pytest
from freezegun import freeze_time

import schedule
from schedule import (
    IntervalError,
    Job,
    Scheduler,
    ScheduleError,
    ScheduleValueError,
)

NOW = datetime.datetime(2024, 1, 1, 12, 0, 0)  # a Monday


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


@pytest.fixture
def sched():
    return Scheduler()


def noop():
    return None


@pytest.mark.parametrize(
    "unit, delta",
    [
        ("seconds", datetime.timedelta(seconds=5)),
        ("minutes", datetime.timedelta(minutes=5)),
        ("hours", datetime.timedelta(hours=5)),
        ("days", datetime.timedelta(days=5)),
        ("weeks", datetime.timedelta(weeks=5)),
    ],
)
def test_every_n_units_sets_next_run_n_units_ahead(frozen, sched, unit, delta):
    job = getattr(sched.every(5), unit).do(noop)
    assert job.unit == unit
    assert job.interval == 5
    assert job.next_run == NOW + delta


@pytest.mark.parametrize(
    "attr, delta",
    [
        ("second", datetime.timedelta(seconds=1)),
        ("minute", datetime.timedelta(minutes=1)),
        ("hour", datetime.timedelta(hours=1)),
        ("day", datetime.timedelta(days=1)),
        ("week", datetime.timedelta(weeks=1)),
    ],
)
def test_every_singular_unit_runs_one_unit_from_now(frozen, sched, attr, delta):
    job = getattr(sched.every(), attr).do(noop)
    assert job.interval == 1
    assert job.next_run == NOW + delta


@pytest.mark.parametrize(
    "attr, plural",
    [
        ("second", "seconds"),
        ("minute", "minutes"),
        ("hour", "hours"),
        ("day", "days"),
        ("week", "weeks"),
    ],
)
def test_singular_unit_with_interval_other_than_one_raises(sched, attr, plural):
    with pytest.raises(IntervalError, match=f"Use {plural} instead of {attr}"):
        getattr(sched.every(2), attr)


@pytest.mark.parametrize(
    "day",
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)
def test_weekday_with_interval_other_than_one_raises(sched, day):
    with pytest.raises(IntervalError):
        getattr(sched.every(2), day)


@pytest.mark.parametrize(
    "day, days_ahead",
    [
        ("monday", 7),  # today is monday and 12:00 is not in the future
        ("tuesday", 1),
        ("wednesday", 2),
        ("thursday", 3),
        ("friday", 4),
        ("saturday", 5),
        ("sunday", 6),
    ],
)
def test_weekday_next_run_lands_on_that_weekday(frozen, sched, day, days_ahead):
    job = getattr(sched.every(), day).do(noop)
    assert job.unit == "weeks"
    assert job.start_day == day
    assert job.next_run == NOW + datetime.timedelta(days=days_ahead)


def test_weekday_with_at_time_sets_time_component(frozen, sched):
    job = sched.every().wednesday.at("13:15").do(noop)
    assert job.next_run == datetime.datetime(2024, 1, 3, 13, 15, 0)


def test_to_picks_interval_inside_inclusive_range(frozen, sched):
    for _ in range(25):
        job = sched.every(5).to(10).seconds.do(noop)
        delta = (job.next_run - NOW).total_seconds()
        assert 5 <= delta <= 10
    assert all(j.latest == 10 for j in sched.jobs)


def test_to_uses_random_randint_between_bounds(frozen, sched, monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 7

    monkeypatch.setattr(schedule.random, "randint", fake_randint)
    job = sched.every(5).to(10).seconds.do(noop)
    assert calls == [(5, 10)]
    assert job.next_run == NOW + datetime.timedelta(seconds=7)


def test_to_with_latest_below_interval_raises(frozen, sched):
    with pytest.raises(ScheduleError):
        sched.every(10).to(5).seconds.do(noop)
    assert sched.jobs == []


def test_tag_collects_unique_hashable_tags_and_returns_job(sched):
    job = sched.every().second
    assert job.tag("a", "b", "a") is job
    job.tag(1, (2, 3))
    assert job.tags == {"a", "b", 1, (2, 3)}


def test_tag_rejects_unhashable(sched):
    with pytest.raises(TypeError):
        sched.every().second.tag(["not", "hashable"])


def test_do_registers_job_and_preserves_function_details(frozen, sched):
    def my_job(a, b=2):
        return a + b

    job = sched.every().second.do(my_job, 1, b=5)
    assert isinstance(job, Job)
    assert sched.jobs == [job]
    assert job.scheduler is sched
    assert job.job_func.__name__ == "my_job"
    assert job.job_func.args == (1,)
    assert job.job_func.keywords == {"b": 5}
    assert job.job_func() == 6


def test_do_without_unit_raises_value_error(sched):
    with pytest.raises(ScheduleValueError):
        sched.every(5).do(noop)
    assert sched.jobs == []


def test_do_without_scheduler_raises_schedule_error(frozen):
    job = Job(5).seconds
    with pytest.raises(ScheduleError):
        job.do(noop)


def test_str_describes_job(frozen, sched):
    def my_job(x):
        pass

    job = sched.every(5).seconds.do(my_job, 1)
    text = str(job)
    assert "interval=5" in text
    assert "unit=seconds" in text
    assert "my_job" in text


def test_repr_variants(frozen, sched):
    plain = sched.every(5).seconds.do(noop)
    text = repr(plain)
    assert text.startswith("Every 5 seconds do noop()")
    assert "last run: [never]" in text
    assert "next run: 2024-01-01 12:00:05" in text

    assert repr(sched.every().hour.do(noop)).startswith("Every 1 hour do noop()")
    assert repr(sched.every().day.at("10:30").do(noop)).startswith(
        "Every 1 day at 10:30:00 do noop()"
    )
    assert repr(sched.every(3).to(6).minutes.do(noop)).startswith(
        "Every 3 to 6 minutes do noop()"
    )
