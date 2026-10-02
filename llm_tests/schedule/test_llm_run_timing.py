"""
Tests for Job.next_run / last_run / should_run computation, actual
execution of due jobs, CancelJob semantics, and until()-based expiry.

Most timing assertions manipulate `next_run`/`cancel_after` directly
(a supported pattern, since these are plain public attributes) so the
suite stays fast and deterministic. A couple of tests use short real
sleeps to verify jobs run "for real" against the wall clock.
"""
import datetime
import time

import pytest

import schedule
from schedule import CancelJob


def noop():
    pass


def make_counter(return_value=None):
    calls = []

    def job():
        calls.append(1)
        return return_value

    job.calls = calls
    return job


# ---------------------------------------------------------------------------
# next_run computation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "interval,unit,lo,hi",
    [
        (1, "seconds", 0.5, 1.5),
        (5, "seconds", 4.0, 6.0),
        (2, "minutes", 110, 130),
    ],
)
def test_next_run_scales_with_interval_and_unit(scheduler, interval, unit, lo, hi):
    now = datetime.datetime.now()
    job = getattr(scheduler.every(interval), unit).do(noop)
    delta = (job.next_run - now).total_seconds()
    assert lo <= delta <= hi


def test_next_run_for_days_with_at_time(scheduler):
    job = scheduler.every().day.at("10:30").do(noop)
    assert (job.next_run.hour, job.next_run.minute, job.next_run.second) == (10, 30, 0)


def test_weekday_next_run_lands_on_requested_weekday_within_a_week(scheduler):
    job = scheduler.every().wednesday.do(noop)
    assert job.next_run.weekday() == 2  # Monday=0 ... Wednesday=2
    now = datetime.datetime.now()
    assert now <= job.next_run <= now + datetime.timedelta(days=7)


# ---------------------------------------------------------------------------
# should_run
# ---------------------------------------------------------------------------

def test_should_run_reflects_next_run_relative_to_now(scheduler):
    job = scheduler.every(10).seconds.do(noop)
    assert job.should_run is False
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    assert job.should_run is True


def test_should_run_raises_assertion_if_next_run_unset(scheduler):
    from schedule import Job

    job = Job(1, scheduler)
    job.unit = "seconds"
    with pytest.raises(AssertionError):
        job.should_run


# ---------------------------------------------------------------------------
# Job.run() behaviour
# ---------------------------------------------------------------------------

def test_run_executes_job_func_updates_last_run_and_reschedules(scheduler):
    job_func = make_counter(return_value="done")
    job = scheduler.every(1).seconds.do(job_func)
    assert job.last_run is None
    first_next_run = job.next_run
    before = datetime.datetime.now()

    result = job.run()

    assert result == "done"
    assert job_func.calls == [1]
    assert job.last_run is not None and job.last_run >= before
    assert job.next_run > first_next_run


# ---------------------------------------------------------------------------
# CancelJob
# ---------------------------------------------------------------------------

def test_job_returning_cancel_job_is_removed_by_scheduler(scheduler):
    def cancel_after_first_run():
        return CancelJob

    job = scheduler.every(1).seconds.do(cancel_after_first_run)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.run_pending()
    assert scheduler.jobs == []


def test_job_not_returning_cancel_job_stays_scheduled(scheduler):
    job = scheduler.every(1).seconds.do(noop)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.run_pending()
    assert job in scheduler.jobs


# ---------------------------------------------------------------------------
# until() / cancel_after expiry
# ---------------------------------------------------------------------------

def test_until_cancels_job_run_after_deadline_passed(scheduler):
    job_func = make_counter()
    job = scheduler.every(1).seconds.do(job_func)
    job.until(datetime.timedelta(seconds=30))
    # Force the deadline into the past without re-triggering until()'s
    # "cannot schedule in the past" guard.
    job.cancel_after = datetime.datetime.now() - datetime.timedelta(seconds=1)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    result = job.run()
    assert result is CancelJob
    assert job_func.calls == []


def test_until_does_not_affect_job_before_deadline(scheduler):
    job_func = make_counter()
    job = scheduler.every(1).seconds.do(job_func)
    job.until(datetime.timedelta(hours=1))
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    result = job.run()
    assert job_func.calls == [1]
    assert result is not CancelJob


def test_until_cancels_when_next_scheduled_run_exceeds_deadline(scheduler):
    # cancel_after is close enough that the *next* run (after this one)
    # would exceed the deadline; run() should return CancelJob even though
    # the job function executes this time.
    job_func = make_counter()
    job = scheduler.every(10).seconds.do(job_func)
    job.cancel_after = datetime.datetime.now() + datetime.timedelta(seconds=2)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    result = job.run()
    assert job_func.calls == [1]  # it did run this time
    assert result is CancelJob  # but won't be scheduled again


# ---------------------------------------------------------------------------
# to() randomized interval
# ---------------------------------------------------------------------------

def test_to_next_run_within_randomized_bounds(scheduler):
    now = datetime.datetime.now()
    job = scheduler.every(1).to(3).seconds.do(noop)
    delta = (job.next_run - now).total_seconds()
    assert 0.5 <= delta <= 3.5


def test_to_latest_smaller_than_interval_raises_on_schedule(scheduler):
    job = scheduler.every(10).to(5).seconds
    with pytest.raises(schedule.ScheduleError):
        job.do(noop)


# ---------------------------------------------------------------------------
# Job ordering (used by run_pending's sorted() call)
# ---------------------------------------------------------------------------

def test_jobs_sort_by_next_run(scheduler):
    later = scheduler.every(100).seconds.do(noop)
    sooner = scheduler.every(1).seconds.do(noop)
    assert sorted([later, sooner]) == [sooner, later]


def test_run_pending_runs_due_jobs_in_next_run_order(scheduler):
    order = []

    def make_job(name):
        def f():
            order.append(name)

        return f

    j_second = scheduler.every(1).seconds.do(make_job("second"))
    j_first = scheduler.every(1).seconds.do(make_job("first"))
    now = datetime.datetime.now()
    j_first.next_run = now - datetime.timedelta(seconds=5)
    j_second.next_run = now - datetime.timedelta(seconds=2)
    scheduler.run_pending()
    assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# Real wall-clock tests: verify jobs actually run when due
# ---------------------------------------------------------------------------

def test_job_actually_runs_after_waiting_for_real_time_to_pass(scheduler):
    job_func = make_counter()
    scheduler.every(1).seconds.do(job_func)
    assert job_func.calls == []
    time.sleep(1.1)
    scheduler.run_pending()
    assert job_func.calls == [1]


def test_job_does_not_run_before_its_interval_elapses(scheduler):
    job_func = make_counter()
    scheduler.every(5).seconds.do(job_func)
    time.sleep(0.2)
    scheduler.run_pending()
    assert job_func.calls == []


def test_job_runs_repeatedly_over_real_time(scheduler):
    job_func = make_counter()
    scheduler.every(1).seconds.do(job_func)
    for _ in range(2):
        time.sleep(1.1)
        scheduler.run_pending()
    assert job_func.calls == [1, 1]
