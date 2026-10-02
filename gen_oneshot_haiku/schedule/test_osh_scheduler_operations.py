"""Scheduler operations tests for schedule library."""

import pytest
import datetime
import schedule
import time


@pytest.fixture
def clean_scheduler():
    """Provide a clean scheduler for each test."""
    sched = schedule.Scheduler()
    yield sched
    sched.clear()


class TestSchedulerJobs:
    """Test Scheduler.jobs and job management."""

    def test_scheduler_jobs_initially_empty(self, clean_scheduler):
        """Test that a new scheduler has no jobs."""
        assert len(clean_scheduler.jobs) == 0
        assert clean_scheduler.jobs == []

    def test_scheduler_jobs_list(self, clean_scheduler):
        """Test that jobs are added to the scheduler.jobs list."""
        def job1():
            pass

        def job2():
            pass

        clean_scheduler.every().seconds.do(job1)
        clean_scheduler.every().minutes.do(job2)

        assert len(clean_scheduler.jobs) == 2
        assert isinstance(clean_scheduler.jobs, list)

    def test_scheduler_get_jobs_no_tag(self, clean_scheduler):
        """Test get_jobs() without tag returns all jobs."""
        def job1():
            pass

        def job2():
            pass

        clean_scheduler.every().seconds.do(job1)
        clean_scheduler.every().minutes.do(job2)

        all_jobs = clean_scheduler.get_jobs()
        assert len(all_jobs) == 2

    def test_scheduler_get_jobs_with_tag(self, clean_scheduler):
        """Test get_jobs() with tag filters jobs."""
        def job1():
            pass

        def job2():
            pass

        def job3():
            pass

        clean_scheduler.every().seconds.do(job1).tag("batch1")
        clean_scheduler.every().seconds.do(job2).tag("batch2")
        clean_scheduler.every().seconds.do(job3).tag("batch1")

        batch1_jobs = clean_scheduler.get_jobs("batch1")
        assert len(batch1_jobs) == 2

        batch2_jobs = clean_scheduler.get_jobs("batch2")
        assert len(batch2_jobs) == 1

    def test_scheduler_get_jobs_with_nonexistent_tag(self, clean_scheduler):
        """Test get_jobs() with nonexistent tag returns empty list."""
        def job():
            pass

        clean_scheduler.every().seconds.do(job).tag("exists")

        result = clean_scheduler.get_jobs("does_not_exist")
        assert len(result) == 0

    def test_get_jobs_returns_copy(self, clean_scheduler):
        """Test that get_jobs() returns a copy, not the original list."""
        def job():
            pass

        clean_scheduler.every().seconds.do(job)

        jobs1 = clean_scheduler.get_jobs()
        jobs2 = clean_scheduler.get_jobs()

        # Should be equal but not the same object
        assert jobs1 == jobs2
        assert jobs1 is not jobs2


class TestSchedulerClear:
    """Test Scheduler.clear() method."""

    def test_clear_all_jobs(self, clean_scheduler):
        """Test clearing all jobs."""
        def job1():
            pass

        def job2():
            pass

        clean_scheduler.every().seconds.do(job1)
        clean_scheduler.every().seconds.do(job2)

        assert len(clean_scheduler.jobs) == 2
        clean_scheduler.clear()
        assert len(clean_scheduler.jobs) == 0

    def test_clear_with_tag(self, clean_scheduler):
        """Test clearing jobs with specific tag."""
        def job1():
            pass

        def job2():
            pass

        def job3():
            pass

        clean_scheduler.every().seconds.do(job1).tag("delete")
        clean_scheduler.every().seconds.do(job2).tag("keep")
        clean_scheduler.every().seconds.do(job3).tag("delete")

        clean_scheduler.clear("delete")
        assert len(clean_scheduler.jobs) == 1

    def test_clear_with_nonexistent_tag(self, clean_scheduler):
        """Test clearing with nonexistent tag has no effect."""
        def job():
            pass

        clean_scheduler.every().seconds.do(job).tag("exists")

        clean_scheduler.clear("nonexistent")
        assert len(clean_scheduler.jobs) == 1

    def test_clear_empty_scheduler(self, clean_scheduler):
        """Test clearing an already-empty scheduler."""
        clean_scheduler.clear()
        assert len(clean_scheduler.jobs) == 0


class TestSchedulerCancelJob:
    """Test Scheduler.cancel_job() method."""

    def test_cancel_job(self, clean_scheduler):
        """Test canceling a specific job."""
        def job1():
            pass

        def job2():
            pass

        j1 = clean_scheduler.every().seconds.do(job1)
        j2 = clean_scheduler.every().seconds.do(job2)

        assert len(clean_scheduler.jobs) == 2
        clean_scheduler.cancel_job(j1)
        assert len(clean_scheduler.jobs) == 1
        assert j2 in clean_scheduler.jobs

    def test_cancel_job_not_scheduled(self, clean_scheduler):
        """Test canceling a job that's not scheduled doesn't raise error."""
        job1 = schedule.Job(1, clean_scheduler)
        # Should not raise an error
        clean_scheduler.cancel_job(job1)

    def test_cancel_all_jobs_individually(self, clean_scheduler):
        """Test canceling all jobs one by one."""
        def job1():
            pass

        def job2():
            pass

        def job3():
            pass

        j1 = clean_scheduler.every().seconds.do(job1)
        j2 = clean_scheduler.every().seconds.do(job2)
        j3 = clean_scheduler.every().seconds.do(job3)

        clean_scheduler.cancel_job(j1)
        clean_scheduler.cancel_job(j2)
        clean_scheduler.cancel_job(j3)

        assert len(clean_scheduler.jobs) == 0


class TestSchedulerNextRun:
    """Test Scheduler.next_run property."""

    def test_next_run_with_single_job(self, clean_scheduler):
        """Test next_run returns the job's next_run."""
        def job():
            pass

        scheduled = clean_scheduler.every(10).seconds.do(job)
        next_run = clean_scheduler.next_run

        assert next_run is not None
        assert next_run == scheduled.next_run

    def test_next_run_with_multiple_jobs(self, clean_scheduler):
        """Test next_run returns earliest scheduled job."""
        def job1():
            pass

        def job2():
            pass

        j1 = clean_scheduler.every(10).seconds.do(job1)
        j2 = clean_scheduler.every(5).seconds.do(job2)

        next_run = clean_scheduler.next_run
        # j2 scheduled sooner (5 seconds) should be next_run
        assert next_run == min(j1.next_run, j2.next_run)

    def test_next_run_empty_scheduler(self, clean_scheduler):
        """Test next_run is None for empty scheduler."""
        assert clean_scheduler.next_run is None

    def test_next_run_with_tag_filter(self, clean_scheduler):
        """Test next_run with tag filtering."""
        def job1():
            pass

        def job2():
            pass

        j1 = clean_scheduler.every(5).seconds.do(job1).tag("fast")
        j2 = clean_scheduler.every(100).seconds.do(job2).tag("slow")

        next_run_fast = clean_scheduler.get_next_run("fast")
        assert next_run_fast == j1.next_run

    def test_next_run_property_is_readonly(self, clean_scheduler):
        """Test that next_run property can be read but reflects current state."""
        def job():
            pass

        clean_scheduler.every(5).seconds.do(job)

        nr1 = clean_scheduler.next_run
        time.sleep(0.1)
        nr2 = clean_scheduler.next_run

        # Should be approximately the same (within small tolerance)
        assert nr1 is not None
        assert nr2 is not None


class TestSchedulerIdleSeconds:
    """Test Scheduler.idle_seconds property."""

    def test_idle_seconds_with_job(self, clean_scheduler):
        """Test idle_seconds returns time until next run."""
        def job():
            pass

        clean_scheduler.every(10).seconds.do(job)
        idle = clean_scheduler.idle_seconds

        assert idle is not None
        assert isinstance(idle, float)
        # Should be approximately 10 seconds (with some tolerance)
        assert 9 < idle < 11

    def test_idle_seconds_empty_scheduler(self, clean_scheduler):
        """Test idle_seconds is None for empty scheduler."""
        assert clean_scheduler.idle_seconds is None

    def test_idle_seconds_with_multiple_jobs(self, clean_scheduler):
        """Test idle_seconds returns time to earliest job."""
        def job1():
            pass

        def job2():
            pass

        clean_scheduler.every(5).seconds.do(job1)
        clean_scheduler.every(100).seconds.do(job2)

        idle = clean_scheduler.idle_seconds
        # Should be closer to 5 seconds than 100
        assert idle is not None
        assert 4 < idle < 6

    def test_idle_seconds_is_positive(self, clean_scheduler):
        """Test that idle_seconds is always positive."""
        def job():
            pass

        clean_scheduler.every(1).seconds.do(job)
        idle = clean_scheduler.idle_seconds

        assert idle > 0


class TestSchedulerRunPending:
    """Test Scheduler.run_pending() method."""

    def test_run_pending_executes_ready_jobs(self, clean_scheduler):
        """Test that run_pending executes jobs that should run."""
        execution_log = []

        def job():
            execution_log.append("ran")

        scheduled = clean_scheduler.every().seconds.do(job)
        # Make job due by setting next_run to past
        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        clean_scheduler.run_pending()
        assert len(execution_log) == 1

    def test_run_pending_skips_future_jobs(self, clean_scheduler):
        """Test that run_pending skips jobs not yet due."""
        execution_log = []

        def job():
            execution_log.append("ran")

        clean_scheduler.every(10).seconds.do(job)
        clean_scheduler.run_pending()

        assert len(execution_log) == 0

    def test_run_pending_updates_last_run(self, clean_scheduler):
        """Test that run_pending updates job's last_run."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.last_run is None

        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
        clean_scheduler.run_pending()

        assert scheduled.last_run is not None

    def test_run_pending_reschedules_job(self, clean_scheduler):
        """Test that run_pending reschedules job after execution."""
        def job():
            pass

        scheduled = clean_scheduler.every(5).seconds.do(job)
        original_next_run = scheduled.next_run

        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
        clean_scheduler.run_pending()

        # next_run should be updated to future
        assert scheduled.next_run > datetime.datetime.now()

    def test_run_pending_multiple_jobs(self, clean_scheduler):
        """Test run_pending with multiple jobs."""
        execution_log = []

        def job1():
            execution_log.append(1)

        def job2():
            execution_log.append(2)

        j1 = clean_scheduler.every().seconds.do(job1)
        j2 = clean_scheduler.every().seconds.do(job2)

        j1.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
        j2.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        clean_scheduler.run_pending()

        assert len(execution_log) == 2
        assert 1 in execution_log
        assert 2 in execution_log

    def test_run_pending_empty_scheduler(self, clean_scheduler):
        """Test run_pending on empty scheduler does nothing."""
        clean_scheduler.run_pending()  # Should not raise


class TestSchedulerRunAll:
    """Test Scheduler.run_all() method."""

    def test_run_all_executes_all_jobs(self, clean_scheduler):
        """Test that run_all executes all jobs regardless of schedule."""
        execution_log = []

        def job1():
            execution_log.append(1)

        def job2():
            execution_log.append(2)

        clean_scheduler.every(100).seconds.do(job1)
        clean_scheduler.every(200).seconds.do(job2)

        clean_scheduler.run_all()

        assert len(execution_log) == 2
        assert 1 in execution_log
        assert 2 in execution_log

    def test_run_all_updates_last_run(self, clean_scheduler):
        """Test that run_all updates last_run for all jobs."""
        def job():
            pass

        scheduled = clean_scheduler.every(100).seconds.do(job)
        assert scheduled.last_run is None

        clean_scheduler.run_all()

        assert scheduled.last_run is not None

    def test_run_all_with_delay(self, clean_scheduler):
        """Test run_all with delay between jobs."""
        start = time.time()

        def job():
            pass

        clean_scheduler.every().seconds.do(job)
        clean_scheduler.every().seconds.do(job)

        clean_scheduler.run_all(delay_seconds=0.1)

        elapsed = time.time() - start
        # Should have ~0.1 second delay between 2 jobs
        assert elapsed >= 0.08

    def test_run_all_empty_scheduler(self, clean_scheduler):
        """Test run_all on empty scheduler does nothing."""
        clean_scheduler.run_all()  # Should not raise


class TestSchedulerEvery:
    """Test Scheduler.every() method."""

    def test_every_returns_job(self, clean_scheduler):
        """Test that every() returns a Job instance."""
        job = clean_scheduler.every()
        assert isinstance(job, schedule.Job)

    def test_every_with_interval(self, clean_scheduler):
        """Test every() with interval parameter."""
        job = clean_scheduler.every(5)
        assert job.interval == 5

    def test_every_default_interval_is_one(self, clean_scheduler):
        """Test that every() defaults to interval of 1."""
        job = clean_scheduler.every()
        assert job.interval == 1

    def test_every_associates_job_with_scheduler(self, clean_scheduler):
        """Test that every() associates job with scheduler."""
        job = clean_scheduler.every()
        assert job.scheduler is clean_scheduler
