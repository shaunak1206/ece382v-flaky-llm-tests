"""Tests for the Scheduler class and for jobs actually running when due."""

import datetime

import pytest
from freezegun import freeze_time

import schedule
from schedule import CancelJob, Job, Scheduler

START = datetime.datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


@pytest.fixture
def sched():
    return Scheduler()


class Recorder:
    """Callable that records how many times and with what it was invoked."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def count(self):
        return len(self.calls)


# ------------------------------------------------------------ run_pending


def test_new_scheduler_has_no_jobs(sched):
    assert sched.jobs == []
    assert sched.get_jobs() == []


def test_job_does_not_run_before_it_is_due(frozen, sched):
    rec = Recorder()
    sched.every(10).seconds.do(rec)
    sched.run_pending()
    assert rec.count == 0
    frozen.move_to("2024-01-01 12:00:09")
    sched.run_pending()
    assert rec.count == 0


def test_job_runs_when_due(frozen, sched):
    rec = Recorder()
    sched.every(10).seconds.do(rec)
    frozen.move_to("2024-01-01 12:00:10")
    sched.run_pending()
    assert rec.count == 1
    # not again until the next interval has elapsed
    sched.run_pending()
    assert rec.count == 1


def test_job_runs_repeatedly_on_its_interval(frozen, sched):
    rec = Recorder()
    sched.every(10).seconds.do(rec)
    for step in range(1, 6):
        frozen.move_to(START + datetime.timedelta(seconds=10 * step))
        sched.run_pending()
        assert rec.count == step


def test_missed_runs_are_not_replayed(frozen, sched):
    rec = Recorder()
    sched.every().minute.do(rec)
    frozen.move_to("2024-01-01 12:30:00")
    sched.run_pending()
    assert rec.count == 1
    assert sched.jobs[0].next_run == datetime.datetime(2024, 1, 1, 12, 31, 0)


def test_run_passes_args_and_kwargs(frozen, sched):
    rec = Recorder()
    sched.every().second.do(rec, 1, 2, key="value")
    frozen.move_to("2024-01-01 12:00:01")
    sched.run_pending()
    assert rec.calls == [((1, 2), {"key": "value"})]


def test_run_updates_last_run_and_next_run(frozen, sched):
    job = sched.every(10).seconds.do(lambda: None)
    assert job.last_run is None
    frozen.move_to("2024-01-01 12:00:10")
    sched.run_pending()
    assert job.last_run == datetime.datetime(2024, 1, 1, 12, 0, 10)
    assert job.next_run == datetime.datetime(2024, 1, 1, 12, 0, 20)


def test_only_due_jobs_run(frozen, sched):
    fast, slow = Recorder(), Recorder()
    sched.every(5).seconds.do(fast)
    sched.every(60).seconds.do(slow)
    frozen.move_to("2024-01-01 12:00:05")
    sched.run_pending()
    assert fast.count == 1
    assert slow.count == 0


def test_due_jobs_run_in_next_run_order(frozen, sched):
    order = []
    sched.every(10).seconds.do(lambda: order.append("ten"))
    sched.every(5).seconds.do(lambda: order.append("five"))
    sched.every(7).seconds.do(lambda: order.append("seven"))
    frozen.move_to("2024-01-01 12:00:10")
    sched.run_pending()
    assert order == ["five", "seven", "ten"]


def test_job_returning_cancel_job_is_removed(frozen, sched):
    rec = Recorder()

    def once():
        rec()
        return CancelJob

    sched.every().second.do(once)
    sched.every().second.do(lambda: CancelJob())
    frozen.move_to("2024-01-01 12:00:01")
    sched.run_pending()
    assert rec.count == 1
    assert sched.jobs == []
    frozen.move_to("2024-01-01 12:00:02")
    sched.run_pending()
    assert rec.count == 1


def test_job_run_returns_job_func_result_and_stays_scheduled(frozen, sched):
    job = sched.every().second.do(lambda: 42)
    assert job.run() == 42
    assert sched.jobs == [job]


def test_exception_in_job_propagates_from_run_pending(frozen, sched):
    def boom():
        raise RuntimeError("boom")

    job = sched.every().second.do(boom)
    frozen.move_to("2024-01-01 12:00:01")
    with pytest.raises(RuntimeError, match="boom"):
        sched.run_pending()
    assert job.last_run is None


def test_should_run_false_then_true(frozen, sched):
    job = sched.every(10).seconds.do(lambda: None)
    assert job.should_run is False
    frozen.move_to("2024-01-01 12:00:10")
    assert job.should_run is True


# --------------------------------------------------------------- run_all


def test_run_all_runs_every_job_regardless_of_schedule(frozen, sched):
    a, b = Recorder(), Recorder()
    sched.every().hour.do(a)
    sched.every().day.do(b)
    sched.run_all()
    assert a.count == 1
    assert b.count == 1


def test_run_all_sleeps_between_jobs(frozen, sched, monkeypatch):
    sleeps = []
    monkeypatch.setattr(schedule.time, "sleep", lambda s: sleeps.append(s))
    for _ in range(3):
        sched.every().day.do(lambda: None)
    sched.run_all(delay_seconds=5)
    assert sleeps == [5, 5, 5]


# ------------------------------------------------- get_jobs / clear / cancel


def test_get_jobs_returns_copy_of_all_jobs(frozen, sched):
    j1 = sched.every().second.do(lambda: None)
    j2 = sched.every().minute.do(lambda: None)
    jobs = sched.get_jobs()
    assert jobs == [j1, j2]
    jobs.clear()
    assert len(sched.jobs) == 2


def test_get_jobs_filters_by_tag(frozen, sched):
    a = sched.every().second.do(lambda: None).tag("alpha")
    b = sched.every().second.do(lambda: None).tag("beta")
    c = sched.every().second.do(lambda: None).tag("alpha", "beta")
    assert sched.get_jobs("alpha") == [a, c]
    assert sched.get_jobs("beta") == [b, c]
    assert sched.get_jobs("missing") == []


def test_clear_removes_all_jobs(frozen, sched):
    for _ in range(3):
        sched.every().second.do(lambda: None)
    sched.clear()
    assert sched.jobs == []


def test_clear_with_tag_removes_only_tagged(frozen, sched):
    sched.every().second.do(lambda: None).tag("daily")
    keep = sched.every().second.do(lambda: None).tag("other")
    untagged = sched.every().second.do(lambda: None)
    sched.clear("daily")
    assert sched.jobs == [keep, untagged]


def test_cancel_job_removes_it_and_it_no_longer_runs(frozen, sched):
    rec = Recorder()
    job = sched.every().second.do(rec)
    other = sched.every().second.do(lambda: None)
    sched.cancel_job(job)
    assert sched.jobs == [other]
    frozen.move_to("2024-01-01 12:00:05")
    sched.run_pending()
    assert rec.count == 0


def test_cancel_job_not_scheduled_does_not_raise(frozen, sched):
    stray = Job(1, sched).seconds
    sched.cancel_job(stray)
    assert sched.jobs == []


# ------------------------------------------------- next_run / idle_seconds


def test_next_run_and_idle_seconds_are_none_without_jobs(sched):
    assert sched.next_run is None
    assert sched.get_next_run() is None
    assert sched.idle_seconds is None


def test_next_run_is_earliest_job(frozen, sched):
    sched.every(30).seconds.do(lambda: None)
    sched.every(10).seconds.do(lambda: None)
    sched.every(20).seconds.do(lambda: None)
    assert sched.next_run == datetime.datetime(2024, 1, 1, 12, 0, 10)


def test_get_next_run_filters_by_tag(frozen, sched):
    sched.every(10).seconds.do(lambda: None).tag("fast")
    sched.every(60).seconds.do(lambda: None).tag("slow")
    assert sched.get_next_run("slow") == datetime.datetime(2024, 1, 1, 12, 1, 0)
    assert sched.get_next_run("nope") is None


def test_idle_seconds_counts_down_to_next_run(frozen, sched):
    sched.every(10).seconds.do(lambda: None)
    assert sched.idle_seconds == 10
    frozen.move_to("2024-01-01 12:00:04")
    assert sched.idle_seconds == 6
    frozen.move_to("2024-01-01 12:00:15")
    assert sched.idle_seconds == -5
