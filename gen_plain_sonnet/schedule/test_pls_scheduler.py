import datetime

import pytest
from freezegun import freeze_time

import schedule
from schedule import CancelJob, Scheduler

NOW = "2024-01-15 12:00:00"


def noop():
    return None


@pytest.fixture
def default_clean():
    schedule.clear()
    yield schedule.default_scheduler
    schedule.clear()


def test_schedulers_do_not_share_jobs():
    a, b = Scheduler(), Scheduler()
    a.every().second.do(noop)
    assert len(a.jobs) == 1
    assert b.jobs == []


def test_run_pending_runs_only_due_jobs():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every(10).seconds.do(calls.append, "fast")
        s.every(1).hours.do(calls.append, "slow")
        s.run_pending()
        assert calls == []
        frozen.move_to("2024-01-15 12:00:10")
        s.run_pending()
    assert calls == ["fast"]


def test_run_pending_runs_in_next_run_order():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every(10).seconds.do(calls.append, "A")
        s.every(5).seconds.do(calls.append, "B")
        frozen.move_to("2024-01-15 12:00:10")
        s.run_pending()
    assert calls == ["B", "A"]


def test_run_pending_reschedules_jobs_after_running():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every(10).seconds.do(calls.append, 1)
        frozen.move_to("2024-01-15 12:00:10")
        s.run_pending()
        s.run_pending()
        assert calls == [1]
        frozen.move_to("2024-01-15 12:00:20")
        s.run_pending()
    assert calls == [1, 1]


def test_run_pending_propagates_job_exception_without_reschedule():
    def boom():
        raise RuntimeError("fail")

    with freeze_time(NOW) as frozen:
        s = Scheduler()
        job = s.every(10).seconds.do(boom)
        frozen.move_to("2024-01-15 12:00:10")
        with pytest.raises(RuntimeError):
            s.run_pending()
        assert job.last_run is None
        assert job.next_run == datetime.datetime(2024, 1, 15, 12, 0, 10)
        assert job in s.jobs


def test_run_all_runs_every_job_even_if_not_due():
    calls = []
    with freeze_time(NOW):
        s = Scheduler()
        s.every().hour.do(calls.append, "a")
        s.every().day.do(calls.append, "b")
        s.run_all()
    assert calls == ["a", "b"]


def test_run_all_sleeps_between_jobs(monkeypatch):
    sleeps = []
    monkeypatch.setattr("schedule.time.sleep", sleeps.append)
    with freeze_time(NOW):
        s = Scheduler()
        s.every().hour.do(noop)
        s.every().day.do(noop)
        s.run_all(delay_seconds=5)
    assert sleeps == [5, 5]


def test_job_returning_cancel_job_class_is_removed():
    with freeze_time(NOW):
        s = Scheduler()
        s.every().hour.do(lambda: CancelJob)
        s.run_all()
    assert s.jobs == []


def test_job_returning_cancel_job_instance_is_removed():
    with freeze_time(NOW):
        s = Scheduler()
        s.every().hour.do(lambda: CancelJob())
        s.run_all()
    assert s.jobs == []


def test_cancel_job_only_removes_that_job():
    with freeze_time(NOW):
        s = Scheduler()
        keep = s.every().hour.do(noop)
        s.every().day.do(lambda: CancelJob)
        s.run_all()
    assert s.jobs == [keep]


def test_cancel_job_removes_scheduled_job():
    s = Scheduler()
    job = s.every().hour.do(noop)
    s.cancel_job(job)
    assert s.jobs == []


def test_cancel_job_ignores_unscheduled_job():
    s = Scheduler()
    keep = s.every().hour.do(noop)
    other = Scheduler().every().hour.do(noop)
    s.cancel_job(other)
    assert s.jobs == [keep]


def test_get_jobs_returns_all_jobs_as_a_copy():
    s = Scheduler()
    job = s.every().hour.do(noop)
    jobs = s.get_jobs()
    assert jobs == [job]
    jobs.clear()
    assert s.jobs == [job]


def test_get_jobs_filters_by_tag():
    s = Scheduler()
    a = s.every().hour.tag("x").do(noop)
    b = s.every().hour.tag("y").do(noop)
    c = s.every().hour.tag("x", "y").do(noop)
    assert s.get_jobs("x") == [a, c]
    assert s.get_jobs("y") == [b, c]
    assert s.get_jobs("missing") == []


def test_clear_removes_all_jobs():
    s = Scheduler()
    s.every().hour.do(noop)
    s.every().day.do(noop)
    s.clear()
    assert s.jobs == []


def test_clear_with_tag_removes_only_tagged_jobs():
    s = Scheduler()
    s.every().hour.tag("x").do(noop)
    keep = s.every().hour.tag("y").do(noop)
    s.every().hour.tag("x", "y").do(noop)
    s.clear("x")
    assert s.jobs == [keep]


def test_next_run_and_idle_seconds_none_without_jobs():
    s = Scheduler()
    assert s.get_next_run() is None
    assert s.next_run is None
    assert s.idle_seconds is None


@freeze_time(NOW)
def test_next_run_is_earliest_of_all_jobs():
    s = Scheduler()
    s.every(30).minutes.do(noop)
    s.every(5).minutes.do(noop)
    assert s.next_run == datetime.datetime(2024, 1, 15, 12, 5, 0)


@freeze_time(NOW)
def test_get_next_run_filters_by_tag():
    s = Scheduler()
    s.every(5).minutes.tag("fast").do(noop)
    s.every(30).minutes.tag("slow").do(noop)
    assert s.get_next_run("slow") == datetime.datetime(2024, 1, 15, 12, 30, 0)
    assert s.get_next_run("nope") is None


@freeze_time(NOW)
def test_idle_seconds_counts_down_to_next_run():
    s = Scheduler()
    s.every(10).seconds.do(noop)
    assert s.idle_seconds == 10.0


def test_job_removed_when_next_run_exceeds_until():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every(10).seconds.until(datetime.datetime(2024, 1, 15, 12, 0, 15)).do(
            calls.append, 1
        )
        frozen.move_to("2024-01-15 12:00:10")
        s.run_pending()
    assert calls == [1]
    assert s.jobs == []


def test_job_not_run_when_already_past_until():
    calls = []
    with freeze_time(NOW) as frozen:
        s = Scheduler()
        s.every(10).seconds.until(datetime.datetime(2024, 1, 15, 12, 0, 15)).do(
            calls.append, 1
        )
        frozen.move_to("2024-01-15 12:00:30")
        s.run_pending()
    assert calls == []
    assert s.jobs == []


def test_repeat_decorator_registers_job_and_keeps_function():
    s = Scheduler()

    @schedule.repeat(s.every(10).seconds, 4, scale=2)
    def compute(a, scale=1):
        return a * scale

    assert compute(3) == 3
    assert len(s.jobs) == 1
    assert s.jobs[0].run() == 8


def test_module_level_functions_use_default_scheduler(default_clean):
    job = schedule.every(10).seconds.tag("t").do(noop)
    assert schedule.default_scheduler.jobs == [job]
    assert schedule.get_jobs("t") == [job]
    schedule.cancel_job(job)
    assert schedule.get_jobs() == []
    assert schedule.next_run() is None
    assert schedule.idle_seconds() is None


def test_module_run_pending_and_run_all(default_clean):
    calls = []
    with freeze_time(NOW) as frozen:
        schedule.every(10).seconds.do(calls.append, "x")
        schedule.run_pending()
        assert calls == []
        frozen.move_to("2024-01-15 12:00:10")
        schedule.run_pending()
        assert calls == ["x"]
        schedule.run_all()
    assert calls == ["x", "x"]
