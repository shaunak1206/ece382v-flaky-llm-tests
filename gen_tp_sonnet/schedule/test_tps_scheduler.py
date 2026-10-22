"""Tests for the Scheduler class, module-level helpers, repeat and exceptions."""
import datetime

import pytest
from freezegun import freeze_time

import schedule

START = "2020-01-01 09:00:00"
T0 = datetime.datetime(2020, 1, 1, 9, 0, 0)


def _idle(scheduler):
    value = scheduler.idle_seconds
    return value() if callable(value) else value


@pytest.fixture(autouse=True)
def _clean_default_scheduler():
    schedule.clear()
    yield
    schedule.clear()


@pytest.fixture
def sched():
    return schedule.Scheduler()


@pytest.fixture
def calls():
    return []


def test_new_scheduler_has_no_jobs(sched):
    assert sched.get_jobs() == []


def test_get_jobs_returns_all_jobs(sched):
    a = sched.every().day.do(lambda: None)
    b = sched.every().hour.do(lambda: None)
    assert set(sched.get_jobs()) == {a, b}


def test_get_jobs_filters_by_tag(sched):
    a = sched.every().day.tag("x").do(lambda: None)
    sched.every().hour.tag("y").do(lambda: None)
    assert sched.get_jobs("x") == [a]


def test_get_jobs_unknown_tag_is_empty(sched):
    sched.every().day.tag("x").do(lambda: None)
    assert sched.get_jobs("missing") == []


def test_clear_removes_all_jobs(sched):
    sched.every().day.do(lambda: None)
    sched.every().hour.do(lambda: None)
    sched.clear()
    assert sched.get_jobs() == []


def test_clear_by_tag_only_removes_tagged(sched):
    sched.every().day.tag("x").do(lambda: None)
    keep = sched.every().hour.tag("y").do(lambda: None)
    sched.clear("x")
    assert sched.get_jobs() == [keep]


def test_cancel_job_removes_only_that_job(sched):
    a = sched.every().day.do(lambda: None)
    b = sched.every().hour.do(lambda: None)
    sched.cancel_job(a)
    assert sched.get_jobs() == [b]


def test_cancel_job_not_scheduled_does_not_raise(sched):
    other = schedule.Scheduler()
    job = other.every().day.do(lambda: None)
    sched.cancel_job(job)
    assert other.get_jobs() == [job]


def test_run_all_runs_every_job(sched, calls):
    sched.every().day.do(calls.append, "day")
    sched.every().hour.do(calls.append, "hour")
    sched.every().monday.do(calls.append, "monday")
    sched.run_all()
    assert sorted(calls) == ["day", "hour", "monday"]


def test_run_pending_only_runs_due_jobs(sched, calls):
    with freeze_time(START) as ft:
        sched.every(10).seconds.do(calls.append, "fast")
        sched.every(60).seconds.do(calls.append, "slow")
        ft.tick(datetime.timedelta(seconds=10))
        sched.run_pending()
    assert calls == ["fast"]


@freeze_time(START)
def test_get_next_run_none_without_jobs(sched):
    assert sched.get_next_run() is None


@freeze_time(START)
def test_get_next_run_returns_earliest(sched):
    sched.every(30).seconds.do(lambda: None)
    sched.every(10).seconds.do(lambda: None)
    sched.every(20).seconds.do(lambda: None)
    assert sched.get_next_run() == T0 + datetime.timedelta(seconds=10)


@freeze_time(START)
def test_get_next_run_filtered_by_tag(sched):
    sched.every(10).seconds.tag("fast").do(lambda: None)
    sched.every(30).seconds.tag("slow").do(lambda: None)
    assert sched.get_next_run("slow") == T0 + datetime.timedelta(seconds=30)


@freeze_time(START)
def test_get_next_run_unknown_tag_is_none(sched):
    sched.every(10).seconds.tag("fast").do(lambda: None)
    assert sched.get_next_run("missing") is None


def test_idle_seconds_none_without_jobs(sched):
    assert _idle(sched) is None


def test_idle_seconds_counts_down(sched):
    with freeze_time(START) as ft:
        sched.every(10).seconds.do(lambda: None)
        assert _idle(sched) == pytest.approx(10.0)
        ft.tick(datetime.timedelta(seconds=4))
        assert _idle(sched) == pytest.approx(6.0)


@freeze_time(START)
def test_idle_seconds_uses_earliest_job(sched):
    sched.every(50).seconds.do(lambda: None)
    sched.every(15).seconds.do(lambda: None)
    assert _idle(sched) == pytest.approx(15.0)


def test_module_every_registers_on_default_scheduler():
    job = schedule.every(10).seconds.do(lambda: None)
    assert schedule.get_jobs() == [job]


def test_module_get_jobs_and_clear_with_tag():
    schedule.every().day.tag("a").do(lambda: None)
    keep = schedule.every().hour.tag("b").do(lambda: None)
    assert len(schedule.get_jobs("a")) == 1
    schedule.clear("a")
    assert schedule.get_jobs() == [keep]


def test_module_clear_removes_everything():
    schedule.every().day.do(lambda: None)
    schedule.clear()
    assert schedule.get_jobs() == []


def test_module_cancel_job():
    job = schedule.every().day.do(lambda: None)
    schedule.cancel_job(job)
    assert schedule.get_jobs() == []


def test_module_run_all(calls):
    schedule.every().day.do(calls.append, 1)
    schedule.every().hour.do(calls.append, 2)
    schedule.run_all()
    assert sorted(calls) == [1, 2]


def test_module_run_pending(calls):
    with freeze_time(START) as ft:
        schedule.every(5).seconds.do(calls.append, "x")
        schedule.run_pending()
        assert calls == []
        ft.tick(datetime.timedelta(seconds=5))
        schedule.run_pending()
    assert calls == ["x"]


def test_module_idle_seconds():
    assert schedule.idle_seconds() is None
    with freeze_time(START):
        schedule.every(10).seconds.do(lambda: None)
        assert schedule.idle_seconds() == pytest.approx(10.0)


@freeze_time(START)
def test_module_next_run():
    assert schedule.next_run() is None
    schedule.every(10).seconds.tag("t").do(lambda: None)
    assert schedule.next_run() == T0 + datetime.timedelta(seconds=10)
    assert schedule.next_run("t") == T0 + datetime.timedelta(seconds=10)


def test_repeat_decorator_schedules_function(calls):
    @schedule.repeat(schedule.every(10).seconds)
    def task():
        calls.append("ran")

    assert len(schedule.get_jobs()) == 1
    schedule.run_all()
    assert calls == ["ran"]


def test_repeat_decorator_passes_arguments(calls):
    @schedule.repeat(schedule.every().hour, "a", key="b")
    def task(first, key=None):
        calls.append((first, key))

    schedule.run_all()
    assert calls == [("a", "b")]


def test_exception_hierarchy():
    assert issubclass(schedule.ScheduleError, Exception)
    assert issubclass(schedule.ScheduleValueError, schedule.ScheduleError)
    assert issubclass(schedule.IntervalError, schedule.ScheduleValueError)
