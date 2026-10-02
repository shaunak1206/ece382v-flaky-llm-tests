"""Tests for the module-level default scheduler API, CancelJob and exceptions."""

import datetime

import pytest
from freezegun import freeze_time

import schedule
from schedule import (
    CancelJob,
    IntervalError,
    Job,
    Scheduler,
    ScheduleError,
    ScheduleValueError,
)


@pytest.fixture(autouse=True)
def clean_default_scheduler():
    schedule.clear()
    yield
    schedule.clear()


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


def test_every_binds_job_to_default_scheduler():
    job = schedule.every(5)
    assert isinstance(job, Job)
    assert job.interval == 5
    assert job.scheduler is schedule.default_scheduler
    assert isinstance(schedule.default_scheduler, Scheduler)
    assert schedule.every().interval == 1


def test_do_registers_on_default_scheduler(frozen):
    job = schedule.every(10).seconds.do(lambda: None)
    assert schedule.jobs is schedule.default_scheduler.jobs
    assert schedule.jobs == [job]
    assert schedule.get_jobs() == [job]


def test_run_pending_runs_due_default_jobs(frozen):
    calls = []
    schedule.every(10).seconds.do(calls.append, "tick")
    schedule.run_pending()
    assert calls == []
    frozen.move_to("2024-01-01 12:00:10")
    schedule.run_pending()
    assert calls == ["tick"]
    schedule.run_pending()
    assert calls == ["tick"]


def test_run_all_runs_default_jobs_immediately(frozen, monkeypatch):
    sleeps = []
    monkeypatch.setattr(schedule.time, "sleep", lambda s: sleeps.append(s))
    calls = []
    schedule.every().day.do(calls.append, "a")
    schedule.every().week.do(calls.append, "b")
    schedule.run_all(delay_seconds=3)
    assert sorted(calls) == ["a", "b"]
    assert sleeps == [3, 3]


def test_get_jobs_by_tag(frozen):
    a = schedule.every().second.do(lambda: None).tag("a")
    schedule.every().second.do(lambda: None).tag("b")
    assert schedule.get_jobs("a") == [a]
    assert len(schedule.get_jobs()) == 2


def test_clear_all_and_by_tag(frozen):
    schedule.every().second.do(lambda: None).tag("drop")
    keep = schedule.every().second.do(lambda: None).tag("keep")
    schedule.clear("drop")
    assert schedule.get_jobs() == [keep]
    # the module-level `jobs` alias must still track the scheduler's list
    assert schedule.jobs == [keep]
    schedule.clear()
    assert schedule.jobs == []


def test_cancel_job(frozen):
    job = schedule.every().second.do(lambda: None)
    schedule.cancel_job(job)
    assert schedule.get_jobs() == []


def test_next_run_default_scheduler(frozen):
    assert schedule.next_run() is None
    schedule.every(30).seconds.do(lambda: None).tag("slow")
    schedule.every(5).seconds.do(lambda: None).tag("fast")
    assert schedule.next_run() == datetime.datetime(2024, 1, 1, 12, 0, 5)
    assert schedule.next_run("slow") == datetime.datetime(2024, 1, 1, 12, 0, 30)
    assert schedule.next_run("none-such") is None


def test_idle_seconds_default_scheduler(frozen):
    assert schedule.idle_seconds() is None
    schedule.every(10).seconds.do(lambda: None)
    assert schedule.idle_seconds() == 10
    frozen.move_to("2024-01-01 12:00:07")
    assert schedule.idle_seconds() == 3


def test_repeat_decorator_schedules_function_and_returns_it(frozen):
    calls = []

    @schedule.repeat(schedule.every(10).seconds, "x", flag=True)
    def work(arg, flag=False):
        calls.append((arg, flag))
        return arg

    assert work.__name__ == "work"
    assert len(schedule.jobs) == 1
    assert calls == []
    frozen.move_to("2024-01-01 12:00:10")
    schedule.run_pending()
    assert calls == [("x", True)]
    assert work("direct") == "direct"


def test_default_scheduler_removes_job_returning_cancel_job(frozen):
    schedule.every().second.do(lambda: CancelJob)
    frozen.move_to("2024-01-01 12:00:01")
    schedule.run_pending()
    assert schedule.get_jobs() == []
    assert isinstance(CancelJob(), CancelJob)


def test_exception_hierarchy():
    assert issubclass(ScheduleError, Exception)
    assert issubclass(ScheduleValueError, ScheduleError)
    assert issubclass(IntervalError, ScheduleValueError)
    assert str(ScheduleValueError("bad value")) == "bad value"


def test_interval_error_raised_and_caught_as_schedule_error():
    with pytest.raises(ScheduleError):
        schedule.every(3).day
    with pytest.raises(IntervalError) as excinfo:
        schedule.every(3).hour
    assert "hours" in str(excinfo.value)


def test_schedule_value_error_for_missing_unit():
    with pytest.raises(ScheduleValueError):
        schedule.every(3).do(lambda: None)
    assert schedule.jobs == []


def test_schedule_error_for_job_without_scheduler(frozen):
    with pytest.raises(ScheduleError):
        Job(1).second.do(lambda: None)
