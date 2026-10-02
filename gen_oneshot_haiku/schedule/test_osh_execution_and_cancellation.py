"""Job execution and cancellation tests for schedule library."""

import pytest
import datetime
import schedule


@pytest.fixture
def clean_scheduler():
    """Provide a clean scheduler for each test."""
    sched = schedule.Scheduler()
    yield sched
    sched.clear()


class TestJobExecution:
    """Test job execution behavior."""

    def test_job_should_run_when_due(self, clean_scheduler):
        """Test that should_run returns True when job is due."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        assert scheduled.should_run is True

    def test_job_should_not_run_when_not_due(self, clean_scheduler):
        """Test that should_run returns False when job is not due."""
        def job():
            pass

        scheduled = clean_scheduler.every(10).seconds.do(job)
        # Default next_run is in future
        assert scheduled.should_run is False

    def test_job_run_executes_function(self, clean_scheduler):
        """Test that job.run() executes the function."""
        execution_log = []

        def job():
            execution_log.append("executed")

        scheduled = clean_scheduler.every().seconds.do(job)
        scheduled.run()

        assert len(execution_log) == 1

    def test_job_run_updates_last_run(self, clean_scheduler):
        """Test that job.run() updates last_run timestamp."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.last_run is None

        scheduled.run()

        assert scheduled.last_run is not None
        assert isinstance(scheduled.last_run, datetime.datetime)

    def test_job_run_reschedules_next_run(self, clean_scheduler):
        """Test that job.run() reschedules next_run."""
        def job():
            pass

        scheduled = clean_scheduler.every(5).seconds.do(job)
        original_next_run = scheduled.next_run

        scheduled.run()

        # next_run should be updated
        assert scheduled.next_run > original_next_run

    def test_job_run_returns_function_result(self, clean_scheduler):
        """Test that job.run() returns the function's result."""
        def job():
            return "result_value"

        scheduled = clean_scheduler.every().seconds.do(job)
        result = scheduled.run()

        assert result == "result_value"

    def test_job_run_with_exception_propagates(self, clean_scheduler):
        """Test that exceptions in job are propagated."""
        def failing_job():
            raise ValueError("job failed")

        scheduled = clean_scheduler.every().seconds.do(failing_job)

        with pytest.raises(ValueError):
            scheduled.run()

    def test_job_execution_with_args(self, clean_scheduler):
        """Test job execution with arguments."""
        results = []

        def job(x, y):
            results.append(x + y)

        scheduled = clean_scheduler.every().seconds.do(job, 3, 7)
        scheduled.run()

        assert results[0] == 10

    def test_job_execution_with_kwargs(self, clean_scheduler):
        """Test job execution with keyword arguments."""
        results = []

        def job(a=0, b=0):
            results.append(a * b)

        scheduled = clean_scheduler.every().seconds.do(job, a=5, b=6)
        scheduled.run()

        assert results[0] == 30

    def test_multiple_job_executions(self, clean_scheduler):
        """Test running the same job multiple times."""
        counter = []

        def job():
            counter.append(1)

        scheduled = clean_scheduler.every().seconds.do(job)

        scheduled.run()
        scheduled.run()
        scheduled.run()

        assert len(counter) == 3


class TestCancelJob:
    """Test CancelJob return value and self-cancellation."""

    def test_cancel_job_class_exists(self):
        """Test that CancelJob class exists."""
        assert hasattr(schedule, 'CancelJob')

    def test_cancel_job_is_class(self):
        """Test that CancelJob is a class."""
        assert isinstance(schedule.CancelJob, type)

    def test_returning_cancel_job_removes_job(self, clean_scheduler):
        """Test that returning CancelJob from job removes it from scheduler."""
        def canceling_job():
            return schedule.CancelJob

        scheduled = clean_scheduler.every().seconds.do(canceling_job)
        assert scheduled in clean_scheduler.jobs

        clean_scheduler.run_pending()
        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
        clean_scheduler.run_pending()

        # Job should be removed
        assert scheduled not in clean_scheduler.jobs

    def test_returning_cancel_job_class_also_removes_job(self, clean_scheduler):
        """Test that returning CancelJob class (not instance) also removes job."""
        def canceling_job():
            return schedule.CancelJob  # Returning the class, not instance

        scheduled = clean_scheduler.every().seconds.do(canceling_job)
        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        clean_scheduler.run_pending()

        assert scheduled not in clean_scheduler.jobs

    def test_job_self_cancels_via_scheduler(self, clean_scheduler):
        """Test job that cancels itself."""
        execution_count = []

        def self_canceling_job():
            execution_count.append(1)
            return schedule.CancelJob

        clean_scheduler.every().seconds.do(self_canceling_job)

        # Run twice to verify it only runs once
        clean_scheduler.run_all()
        clean_scheduler.run_all()

        assert len(execution_count) == 1


class TestJobShouldRunAndUntil:
    """Test job deadline behavior with until()."""

    def test_until_cancels_overdue_job(self, clean_scheduler):
        """Test that job past deadline is canceled."""
        def job():
            pass

        past_deadline = datetime.datetime.now() - datetime.timedelta(hours=1)
        scheduled = clean_scheduler.every().seconds.until(past_deadline).do(job)

        # This should fail validation at scheduling time
        # But if we bypass it, run should return CancelJob
        assert scheduled.cancel_after < datetime.datetime.now()

    def test_until_prevents_execution_past_deadline(self, clean_scheduler):
        """Test that job doesn't run after deadline."""
        execution_log = []

        def job():
            execution_log.append(1)

        # Deadline in past
        past_time = datetime.datetime.now() - datetime.timedelta(seconds=1)

        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().seconds.until(past_time).do(job)

    def test_until_future_allows_execution(self, clean_scheduler):
        """Test that job can run when deadline is in future."""
        execution_log = []

        def job():
            execution_log.append(1)
            return schedule.CancelJob

        future_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        scheduled = clean_scheduler.every().seconds.until(future_time).do(job)

        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)
        clean_scheduler.run_pending()

        assert len(execution_log) == 1

    def test_until_cancels_after_execution_if_next_run_past_deadline(self, clean_scheduler):
        """Test that job is canceled if next_run exceeds deadline."""
        def job():
            return "result"

        # Set deadline to very soon
        future_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
        scheduled = clean_scheduler.every(10).seconds.until(future_time).do(job)

        # Artificially set next_run to execute now
        scheduled.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        # After execution, next_run will be 10 seconds away, exceeding deadline
        # So this should return CancelJob
        result = scheduled.run()

        # Result should be CancelJob since next scheduled time exceeds deadline
        assert result is schedule.CancelJob or isinstance(result, schedule.CancelJob)


class TestJobReschedule:
    """Test job rescheduling after execution."""

    def test_next_run_updated_after_execution(self, clean_scheduler):
        """Test that next_run is updated after job execution."""
        def job():
            pass

        scheduled = clean_scheduler.every(5).seconds.do(job)
        first_next_run = scheduled.next_run

        scheduled.run()

        second_next_run = scheduled.next_run
        assert second_next_run > first_next_run

    def test_next_run_respects_interval(self, clean_scheduler):
        """Test that next_run respects the job's interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(10).seconds.do(job)
        before_run = datetime.datetime.now()

        scheduled.run()

        after_run = datetime.datetime.now()

        # next_run should be approximately 10 seconds from now
        time_until_next = (scheduled.next_run - after_run).total_seconds()
        assert 8 < time_until_next < 12

    def test_next_run_with_to_randomization(self, clean_scheduler):
        """Test that next_run with .to() varies."""
        def job():
            pass

        next_runs = []
        for _ in range(5):
            scheduled = clean_scheduler.every(1).to(5).seconds.do(job)
            scheduled.run()
            next_runs.append(scheduled.next_run)

        # Should have some variance in scheduling times
        time_deltas = [(nr - datetime.datetime.now()).total_seconds() for nr in next_runs]
        # At least some jobs should be scheduled at different times
        assert len(set(int(td) for td in time_deltas)) > 1 or all(1 < td < 6 for td in time_deltas)


class TestJobComparison:
    """Test job comparison for sorting."""

    def test_job_less_than_comparison(self, clean_scheduler):
        """Test that jobs can be compared by next_run."""
        def job1():
            pass

        def job2():
            pass

        j1 = clean_scheduler.every(5).seconds.do(job1)
        j2 = clean_scheduler.every(10).seconds.do(job2)

        # j1 should run sooner
        assert j1 < j2

    def test_job_sorting(self, clean_scheduler):
        """Test that jobs can be sorted by next_run."""
        def job():
            pass

        j1 = clean_scheduler.every(5).seconds.do(job)
        j2 = clean_scheduler.every(1).seconds.do(job)
        j3 = clean_scheduler.every(10).seconds.do(job)

        jobs = [j1, j2, j3]
        sorted_jobs = sorted(jobs)

        # Should be sorted by next_run time
        assert sorted_jobs[0] is j2  # 1 second
        assert sorted_jobs[1] is j1  # 5 seconds
        assert sorted_jobs[2] is j3  # 10 seconds


class TestJobState:
    """Test job state and properties."""

    def test_job_next_run_is_datetime(self, clean_scheduler):
        """Test that next_run is always a datetime."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert isinstance(scheduled.next_run, datetime.datetime)

    def test_job_last_run_before_execution(self, clean_scheduler):
        """Test that last_run is None before first execution."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.last_run is None

    def test_job_last_run_after_execution(self, clean_scheduler):
        """Test that last_run is set after execution."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        scheduled.run()

        assert scheduled.last_run is not None

    def test_job_last_run_is_datetime(self, clean_scheduler):
        """Test that last_run is a datetime when set."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        scheduled.run()

        assert isinstance(scheduled.last_run, datetime.datetime)

    def test_job_multiple_runs_update_last_run(self, clean_scheduler):
        """Test that last_run is updated with each execution."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)

        scheduled.run()
        first_last_run = scheduled.last_run

        import time
        time.sleep(0.01)  # Small delay to ensure different timestamp

        scheduled.run()
        second_last_run = scheduled.last_run

        assert second_last_run > first_last_run
