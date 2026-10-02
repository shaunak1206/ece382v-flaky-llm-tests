"""
Tests for the Scheduler class (run_pending, run_all, clear, cancel_job,
get_jobs, next_run, idle_seconds) and the module-level default-scheduler
convenience functions.
"""
import datetime

import schedule
from schedule import CancelJob, Scheduler


def noop():
    pass


def make_counter():
    calls = []

    def job():
        calls.append(1)

    job.calls = calls
    return job


# ---------------------------------------------------------------------------
# basic bookkeeping
# ---------------------------------------------------------------------------

def test_new_scheduler_has_no_jobs(scheduler):
    assert scheduler.jobs == []


def test_do_appends_multiple_jobs_to_jobs_list(scheduler):
    scheduler.every().seconds  # not yet registered
    assert scheduler.jobs == []
    j1 = scheduler.every().seconds.do(noop)
    j2 = scheduler.every().minutes.do(noop)
    assert scheduler.jobs == [j1, j2]


# ---------------------------------------------------------------------------
# get_jobs / clear / cancel_job
# ---------------------------------------------------------------------------

def test_get_jobs_returns_copy_and_filters_by_tag(scheduler):
    j1 = scheduler.every().seconds.do(noop).tag("a")
    scheduler.every().seconds.do(noop).tag("b")
    assert scheduler.get_jobs("a") == [j1]
    assert scheduler.get_jobs("nonexistent") == []
    all_jobs = scheduler.get_jobs()
    all_jobs.clear()
    assert len(scheduler.jobs) == 2  # mutating the returned list didn't affect state


def test_clear_removes_all_jobs(scheduler):
    scheduler.every().seconds.do(noop)
    scheduler.every().minutes.do(noop)
    scheduler.clear()
    assert scheduler.jobs == []


def test_clear_by_tag_removes_only_matching(scheduler):
    keep = scheduler.every().seconds.do(noop).tag("keep")
    scheduler.every().seconds.do(noop).tag("drop")
    scheduler.clear("drop")
    assert scheduler.jobs == [keep]


def test_cancel_job_removes_it_and_is_idempotent(scheduler):
    job = scheduler.every().seconds.do(noop)
    scheduler.cancel_job(job)
    assert job not in scheduler.jobs
    scheduler.cancel_job(job)  # cancelling again should not raise


# ---------------------------------------------------------------------------
# next_run / idle_seconds
# ---------------------------------------------------------------------------

def test_next_run_returns_the_earliest_job(scheduler):
    assert scheduler.next_run is None
    far = scheduler.every(10).seconds.do(noop)
    near = scheduler.every(1).seconds.do(noop)
    assert scheduler.next_run == near.next_run
    assert scheduler.next_run < far.next_run


def test_get_next_run_with_tag_filter(scheduler):
    tagged = scheduler.every(1).seconds.do(noop).tag("x")
    scheduler.every(1).seconds.do(noop)
    tagged.next_run = datetime.datetime.now() - datetime.timedelta(seconds=5)
    assert scheduler.get_next_run("x") == tagged.next_run
    assert scheduler.get_next_run("nope") is None


def test_idle_seconds_none_when_empty_positive_and_negative(scheduler):
    assert scheduler.idle_seconds is None
    job = scheduler.every(10).seconds.do(noop)
    assert scheduler.idle_seconds > 0
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    assert scheduler.idle_seconds < 0


# ---------------------------------------------------------------------------
# run_pending
# ---------------------------------------------------------------------------

def test_run_pending_runs_only_due_jobs(scheduler):
    due_func, future_func = make_counter(), make_counter()
    due = scheduler.every(1).seconds.do(due_func)
    scheduler.every(10).seconds.do(future_func)
    due.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.run_pending()
    assert due_func.calls == [1]
    assert future_func.calls == []


def test_run_pending_removes_job_that_returns_cancel(scheduler):
    def cancel_me():
        return CancelJob

    job = scheduler.every(1).seconds.do(cancel_me)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    scheduler.run_pending()
    assert job not in scheduler.jobs


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------

def test_run_all_runs_every_job_once_regardless_of_due_time(scheduler):
    f1, f2, f3 = make_counter(), make_counter(), make_counter()
    scheduler.every(100).seconds.do(f1)  # far in the future
    scheduler.every().minutes.do(f2)
    scheduler.every().hours.do(f3)
    scheduler.run_all()
    assert f1.calls == f2.calls == f3.calls == [1]


def test_run_all_with_delay_waits_between_jobs(scheduler):
    f1, f2 = make_counter(), make_counter()
    scheduler.every().seconds.do(f1)
    scheduler.every().seconds.do(f2)
    start = datetime.datetime.now()
    scheduler.run_all(delay_seconds=0.2)
    elapsed = (datetime.datetime.now() - start).total_seconds()
    assert f1.calls == [1] and f2.calls == [1]
    assert elapsed >= 0.2


# ---------------------------------------------------------------------------
# module-level default-scheduler shortcuts
# ---------------------------------------------------------------------------

def test_module_every_and_get_jobs_use_default_scheduler():
    job = schedule.every().seconds.do(noop).tag("mod")
    assert job in schedule.default_scheduler.jobs
    assert schedule.get_jobs("mod") == [job]


def test_module_clear_and_cancel_job():
    job = schedule.every().seconds.do(noop)
    schedule.clear()
    assert schedule.get_jobs() == []
    job2 = schedule.every().seconds.do(noop)
    schedule.cancel_job(job2)
    assert job2 not in schedule.get_jobs()


def test_module_next_run_and_idle_seconds():
    job = schedule.every(5).seconds.do(noop)
    assert schedule.next_run() == job.next_run
    assert schedule.idle_seconds() > 0


def test_module_run_pending_executes_due_job():
    job_func = make_counter()
    job = schedule.every(1).seconds.do(job_func)
    job.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
    schedule.run_pending()
    assert job_func.calls == [1]


def test_module_run_all_executes_job_and_jobs_alias_stays_in_sync():
    job_func = make_counter()
    schedule.every(1).hours.do(job_func)
    schedule.run_all()
    assert job_func.calls == [1]
    assert schedule.jobs == schedule.default_scheduler.jobs


# ---------------------------------------------------------------------------
# repeat() decorator
# ---------------------------------------------------------------------------

def test_repeat_decorator_schedules_and_passes_args():
    received = []

    @schedule.repeat(schedule.every(1).seconds, "hello", who="world")
    def decorated(greeting, who=None):
        received.append((greeting, who))

    jobs = schedule.get_jobs()
    assert len(jobs) == 1
    schedule.run_all()
    assert received == [("hello", "world")]


def test_scheduler_instances_are_independent():
    s1 = Scheduler()
    s2 = Scheduler()
    s1.every().seconds.do(noop)
    assert s1.jobs != s2.jobs
    assert s2.jobs == []
