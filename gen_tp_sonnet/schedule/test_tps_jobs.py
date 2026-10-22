"""Tests for Job behaviour: do, run, tags, to, until and CancelJob."""
import datetime

import pytest
from freezegun import freeze_time

import schedule

START = "2020-01-01 00:00:00"
T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)


def _is_cancel(value):
    return value is schedule.CancelJob or isinstance(value, schedule.CancelJob)


@pytest.fixture
def sched():
    return schedule.Scheduler()


@pytest.fixture
def calls():
    return []


def test_do_returns_job_and_registers_it(sched):
    job = sched.every(10).seconds.do(lambda: None)
    assert isinstance(job, schedule.Job)
    assert sched.get_jobs() == [job]


def test_job_not_registered_before_do(sched):
    sched.every(10).seconds
    assert sched.get_jobs() == []


def test_do_passes_args_and_kwargs(sched, calls):
    def func(a, b, c=None):
        calls.append((a, b, c))

    sched.every().second.do(func, 1, 2, c=3)
    sched.run_all()
    assert calls == [(1, 2, 3)]


@freeze_time(START)
def test_job_does_not_run_before_interval(sched, calls):
    sched.every(10).seconds.do(calls.append, "x")
    with freeze_time(T0 + datetime.timedelta(seconds=9)):
        sched.run_pending()
    assert calls == []


@freeze_time(START)
def test_job_runs_when_interval_elapsed(sched, calls):
    sched.every(10).seconds.do(calls.append, "x")
    with freeze_time(T0 + datetime.timedelta(seconds=10)):
        sched.run_pending()
    assert calls == ["x"]


def test_job_runs_repeatedly_each_interval(sched, calls):
    with freeze_time(START) as ft:
        sched.every(10).seconds.do(calls.append, "x")
        for _ in range(3):
            ft.tick(datetime.timedelta(seconds=10))
            sched.run_pending()
    assert len(calls) == 3


def test_run_pending_does_not_catch_up_missed_runs(sched, calls):
    with freeze_time(START) as ft:
        sched.every().minute.do(calls.append, "x")
        ft.tick(datetime.timedelta(hours=1))
        sched.run_pending()
        sched.run_pending()
    assert len(calls) == 1


def test_should_run_transitions(sched):
    with freeze_time(START) as ft:
        job = sched.every(10).seconds.do(lambda: None)
        assert job.should_run is False or job.should_run() is False
        ft.tick(datetime.timedelta(seconds=10))
        assert job.should_run is True or job.should_run() is True


def test_run_returns_job_function_value(sched):
    with freeze_time(START):
        job = sched.every(10).seconds.do(lambda: 42)
        assert job.run() == 42


def test_run_updates_last_run_and_next_run(sched):
    with freeze_time(START) as ft:
        job = sched.every(10).seconds.do(lambda: None)
        assert job.last_run is None
        ft.tick(datetime.timedelta(seconds=10))
        job.run()
        now = T0 + datetime.timedelta(seconds=10)
        assert job.last_run == now
        assert job.next_run == now + datetime.timedelta(seconds=10)


def test_run_returns_cancel_job_when_function_returns_it(sched):
    job = sched.every(10).seconds.do(lambda: schedule.CancelJob)
    assert _is_cancel(job.run())


def test_run_pending_removes_job_returning_cancel_job(sched, calls):
    def once():
        calls.append(1)
        return schedule.CancelJob

    with freeze_time(START) as ft:
        sched.every(1).seconds.do(once)
        ft.tick(datetime.timedelta(seconds=1))
        sched.run_pending()
        ft.tick(datetime.timedelta(seconds=1))
        sched.run_pending()
    assert calls == [1]
    assert sched.get_jobs() == []


def test_run_pending_propagates_job_exceptions(sched):
    def boom():
        raise RuntimeError("boom")

    with freeze_time(START) as ft:
        sched.every(1).seconds.do(boom)
        ft.tick(datetime.timedelta(seconds=1))
        with pytest.raises(RuntimeError):
            sched.run_pending()


def test_tag_returns_same_job():
    job = schedule.Scheduler().every().day
    assert job.tag("a") is job


def test_tag_discards_duplicates(sched):
    job = sched.every().day.tag("a", "a", "b").do(lambda: None)
    assert len(job.tags) == 2
    assert set(job.tags) == {"a", "b"}


def test_tag_accumulates_across_calls(sched):
    job = sched.every().day.tag("a").tag("b").do(lambda: None)
    assert set(job.tags) == {"a", "b"}


def test_tag_rejects_unhashable(sched):
    with pytest.raises(TypeError):
        sched.every().day.tag(["not", "hashable"])


def test_to_initial_next_run_within_range(sched):
    with freeze_time(START):
        for _ in range(25):
            job = sched.every(5).to(10).seconds.do(lambda: None)
            delta = job.next_run - T0
            assert datetime.timedelta(seconds=5) <= delta <= datetime.timedelta(seconds=10)


def test_to_rescheduled_next_run_within_range(sched):
    with freeze_time(START) as ft:
        job = sched.every(5).to(10).seconds.do(lambda: None)
        for _ in range(25):
            ft.tick(datetime.timedelta(seconds=10))
            job.run()
            delta = job.next_run - datetime.datetime.now()
            assert datetime.timedelta(seconds=5) <= delta <= datetime.timedelta(seconds=10)


@freeze_time(START)
def test_until_past_datetime_raises(sched):
    past = T0 - datetime.timedelta(seconds=1)
    with pytest.raises(schedule.ScheduleValueError):
        sched.every().second.until(past).do(lambda: None)


def test_until_datetime_cancels_job_once_next_run_passes_deadline(sched, calls):
    with freeze_time(START) as ft:
        deadline = T0 + datetime.timedelta(seconds=25)
        sched.every(10).seconds.until(deadline).do(calls.append, "x")
        for _ in range(5):
            ft.tick(datetime.timedelta(seconds=10))
            sched.run_pending()
    assert len(calls) == 2
    assert sched.get_jobs() == []


def test_until_timedelta(sched, calls):
    with freeze_time(START) as ft:
        sched.every(10).seconds.until(datetime.timedelta(seconds=25)).do(calls.append, "x")
        for _ in range(5):
            ft.tick(datetime.timedelta(seconds=10))
            sched.run_pending()
    assert len(calls) == 2
    assert sched.get_jobs() == []


def test_until_time_of_day_string(sched, calls):
    with freeze_time(START) as ft:
        sched.every(10).seconds.until("00:00:25").do(calls.append, "x")
        for _ in range(5):
            ft.tick(datetime.timedelta(seconds=10))
            sched.run_pending()
    assert len(calls) == 2


def test_until_time_object(sched, calls):
    with freeze_time("2020-01-01 09:00:00") as ft:
        sched.every(1).hours.until(datetime.time(11, 30)).do(calls.append, "x")
        for _ in range(5):
            ft.tick(datetime.timedelta(hours=1))
            sched.run_pending()
    assert len(calls) == 2
    assert sched.get_jobs() == []


@freeze_time(START)
def test_until_invalid_string_raises(sched):
    with pytest.raises(schedule.ScheduleValueError):
        sched.every().second.until("not a time")


@freeze_time(START)
def test_until_invalid_type_raises(sched):
    with pytest.raises(TypeError):
        sched.every().second.until(12345)


def test_run_after_deadline_returns_cancel_job_without_calling(sched, calls):
    with freeze_time(START) as ft:
        job = sched.every(10).seconds.until(
            T0 + datetime.timedelta(seconds=15)
        ).do(calls.append, "x")
        ft.tick(datetime.timedelta(seconds=20))
        result = job.run()
    assert _is_cancel(result)
    assert calls == []


def test_daily_at_job_reschedules_for_next_day(sched, calls):
    with freeze_time("2020-01-01 10:00:00") as ft:
        job = sched.every().day.at("10:30").do(calls.append, "x")
        assert job.next_run == datetime.datetime(2020, 1, 1, 10, 30)
        ft.move_to("2020-01-01 10:30:00")
        sched.run_pending()
        assert calls == ["x"]
        assert job.next_run == datetime.datetime(2020, 1, 2, 10, 30)
