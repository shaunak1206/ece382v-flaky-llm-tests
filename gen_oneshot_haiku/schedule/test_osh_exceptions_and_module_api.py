"""Exception and module-level API tests for schedule library."""

import pytest
import datetime
import schedule


class TestExceptions:
    """Test schedule exception types."""

    def test_schedule_error_exists(self):
        """Test that ScheduleError exception exists."""
        assert issubclass(schedule.ScheduleError, Exception)

    def test_schedule_value_error_exists(self):
        """Test that ScheduleValueError exception exists."""
        assert issubclass(schedule.ScheduleValueError, schedule.ScheduleError)

    def test_interval_error_exists(self):
        """Test that IntervalError exception exists."""
        assert issubclass(schedule.IntervalError, schedule.ScheduleValueError)

    def test_schedule_error_raised_for_invalid_unit(self):
        """Test that ScheduleError is raised for invalid unit."""
        sched = schedule.Scheduler()

        job = schedule.Job(1, sched)
        job.unit = "invalid_unit"

        with pytest.raises(schedule.ScheduleValueError):
            job._schedule_next_run()

    def test_interval_error_raised_for_invalid_singular_use(self):
        """Test that IntervalError is raised for invalid singular unit use."""
        sched = schedule.Scheduler()

        with pytest.raises(schedule.IntervalError):
            sched.every(2).second

    def test_schedule_error_raised_when_no_scheduler(self):
        """Test that ScheduleError is raised when job has no scheduler."""
        job = schedule.Job(1, scheduler=None)

        with pytest.raises(schedule.ScheduleError):
            job.do(lambda: None)

    def test_schedule_value_error_raised_for_invalid_at_time(self):
        """Test that ScheduleValueError is raised for invalid at() time."""
        sched = schedule.Scheduler()

        with pytest.raises(schedule.ScheduleValueError):
            sched.every().day.at("25:00").do(lambda: None)

    def test_type_error_raised_for_non_string_at(self):
        """Test that TypeError is raised for non-string at() value."""
        sched = schedule.Scheduler()

        with pytest.raises(TypeError):
            sched.every().day.at(1030).do(lambda: None)


class TestModuleLevelApi:
    """Test module-level convenience functions."""

    def test_module_level_every_function(self):
        """Test the module-level every() function."""
        def job():
            pass

        scheduled = schedule.every(5).seconds.do(job)
        assert scheduled is not None
        assert isinstance(scheduled, schedule.Job)

    def test_module_level_every_uses_default_scheduler(self):
        """Test that module-level every() uses default scheduler."""
        assert hasattr(schedule, 'default_scheduler')
        assert isinstance(schedule.default_scheduler, schedule.Scheduler)

        # Clear default scheduler first
        schedule.clear()

        def job():
            pass

        schedule.every().seconds.do(job)
        assert len(schedule.default_scheduler.jobs) == 1

    def test_module_level_run_pending(self):
        """Test the module-level run_pending() function."""
        schedule.clear()
        execution_log = []

        def job():
            execution_log.append(1)

        j = schedule.every().seconds.do(job)
        j.next_run = datetime.datetime.now() - datetime.timedelta(seconds=1)

        schedule.run_pending()
        assert len(execution_log) == 1

    def test_module_level_run_all(self):
        """Test the module-level run_all() function."""
        schedule.clear()
        execution_log = []

        def job():
            execution_log.append(1)

        schedule.every(100).seconds.do(job)

        schedule.run_all()
        assert len(execution_log) == 1

    def test_module_level_run_all_with_delay(self):
        """Test the module-level run_all() with delay parameter."""
        import time
        schedule.clear()

        def job():
            pass

        schedule.every().seconds.do(job)
        schedule.every().seconds.do(job)

        start = time.time()
        schedule.run_all(delay_seconds=0.05)
        elapsed = time.time() - start

        assert elapsed >= 0.04

    def test_module_level_get_jobs(self):
        """Test the module-level get_jobs() function."""
        schedule.clear()

        def job1():
            pass

        def job2():
            pass

        schedule.every().seconds.do(job1).tag("tag1")
        schedule.every().seconds.do(job2).tag("tag2")

        jobs = schedule.get_jobs()
        assert len(jobs) == 2

    def test_module_level_get_jobs_with_tag(self):
        """Test the module-level get_jobs() with tag parameter."""
        schedule.clear()

        def job():
            pass

        schedule.every().seconds.do(job).tag("filter_me")
        schedule.every().seconds.do(job).tag("other")

        jobs = schedule.get_jobs("filter_me")
        assert len(jobs) == 1

    def test_module_level_clear(self):
        """Test the module-level clear() function."""
        schedule.clear()

        def job():
            pass

        schedule.every().seconds.do(job)
        assert len(schedule.get_jobs()) == 1

        schedule.clear()
        assert len(schedule.get_jobs()) == 0

    def test_module_level_clear_with_tag(self):
        """Test the module-level clear() with tag parameter."""
        schedule.clear()

        def job():
            pass

        schedule.every().seconds.do(job).tag("delete")
        schedule.every().seconds.do(job).tag("keep")

        schedule.clear("delete")
        assert len(schedule.get_jobs()) == 1

    def test_module_level_cancel_job(self):
        """Test the module-level cancel_job() function."""
        schedule.clear()

        def job():
            pass

        j = schedule.every().seconds.do(job)
        assert len(schedule.get_jobs()) == 1

        schedule.cancel_job(j)
        assert len(schedule.get_jobs()) == 0

    def test_module_level_next_run(self):
        """Test the module-level next_run() function."""
        schedule.clear()

        def job():
            pass

        schedule.every(10).seconds.do(job)

        nr = schedule.next_run()
        assert nr is not None
        assert isinstance(nr, datetime.datetime)

    def test_module_level_idle_seconds(self):
        """Test the module-level idle_seconds() function."""
        schedule.clear()

        def job():
            pass

        schedule.every(10).seconds.do(job)

        idle = schedule.idle_seconds()
        assert idle is not None
        assert isinstance(idle, float)
        assert 9 < idle < 11

    def test_module_level_jobs_reference(self):
        """Test the module-level jobs reference."""
        schedule.clear()

        assert hasattr(schedule, 'jobs')
        assert isinstance(schedule.jobs, list)
        assert schedule.jobs == schedule.default_scheduler.jobs

    def test_module_level_repeat_decorator(self):
        """Test the module-level repeat() decorator."""
        schedule.clear()

        job = schedule.every(5).seconds

        @schedule.repeat(job)
        def my_job():
            return "executed"

        assert my_job() == "executed"
        assert len(schedule.get_jobs()) == 1


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_zero_interval(self):
        """Test scheduling with zero interval."""
        sched = schedule.Scheduler()

        def job():
            pass

        # Zero interval should still work (though unusual)
        j = sched.every(0).seconds.do(job)
        assert j.interval == 0

    def test_very_large_interval(self):
        """Test scheduling with very large interval."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every(999999).days.do(job)
        assert j.interval == 999999

    def test_job_with_no_arguments(self):
        """Test scheduling a function that takes no arguments."""
        sched = schedule.Scheduler()

        def job():
            return "no args"

        j = sched.every().seconds.do(job)
        assert j.job_func() == "no args"

    def test_job_with_many_arguments(self):
        """Test scheduling a function with many arguments."""
        sched = schedule.Scheduler()

        def job(a, b, c, d, e):
            return a + b + c + d + e

        j = sched.every().seconds.do(job, 1, 2, 3, 4, 5)
        assert j.job_func() == 15

    def test_job_with_lambda(self):
        """Test scheduling a lambda function."""
        sched = schedule.Scheduler()

        j = sched.every().seconds.do(lambda: "lambda result")
        assert j.job_func() == "lambda result"

    def test_job_that_returns_none(self):
        """Test scheduling a job that returns None."""
        sched = schedule.Scheduler()
        execution_log = []

        def job():
            execution_log.append(1)
            return None

        j = sched.every().seconds.do(job)
        result = j.run()

        assert result is None
        assert len(execution_log) == 1

    def test_job_that_returns_false(self):
        """Test scheduling a job that returns False."""
        sched = schedule.Scheduler()

        def job():
            return False

        j = sched.every().seconds.do(job)
        result = j.run()

        assert result is False

    def test_empty_tag_set(self):
        """Test job with no tags."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every().seconds.do(job)
        assert len(j.tags) == 0
        assert isinstance(j.tags, set)

    def test_max_timeout_in_at(self):
        """Test .at() with 23:59:59."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every().day.at("23:59:59").do(job)
        assert j.at_time.hour == 23
        assert j.at_time.minute == 59
        assert j.at_time.second == 59

    def test_min_timeout_in_at(self):
        """Test .at() with 00:00:00."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every().day.at("00:00:00").do(job)
        assert j.at_time.hour == 0
        assert j.at_time.minute == 0
        assert j.at_time.second == 0

    def test_scheduler_with_many_jobs(self):
        """Test scheduler with many jobs."""
        sched = schedule.Scheduler()

        def job():
            pass

        for i in range(100):
            sched.every(i + 1).seconds.do(job).tag(f"job_{i}")

        assert len(sched.jobs) == 100

    def test_reusable_job_function(self):
        """Test using the same function for multiple scheduled jobs."""
        sched = schedule.Scheduler()

        execution_log = []

        def job():
            execution_log.append(1)

        j1 = sched.every(5).seconds.do(job)
        j2 = sched.every(10).seconds.do(job)

        assert len(sched.jobs) == 2

        j1.run()
        assert len(execution_log) == 1

        j2.run()
        assert len(execution_log) == 2

    def test_callable_class_instance(self):
        """Test scheduling a callable class instance."""
        sched = schedule.Scheduler()

        class CallableJob:
            def __init__(self):
                self.call_count = 0

            def __call__(self):
                self.call_count += 1
                return self.call_count

        job_obj = CallableJob()
        j = sched.every().seconds.do(job_obj)

        assert j.job_func() == 1
        assert j.job_func() == 2


class TestJobString:
    """Test job string representations."""

    def test_job_str_with_seconds(self):
        """Test __str__ for seconds job."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every(5).seconds.do(job)
        job_str = str(j)

        assert "5" in job_str
        assert "second" in job_str

    def test_job_repr_with_intervals(self):
        """Test __repr__ includes interval information."""
        sched = schedule.Scheduler()

        def my_job():
            pass

        j = sched.every(10).minutes.do(my_job)
        job_repr = repr(j)

        assert "10" in job_repr
        assert "minute" in job_repr

    def test_job_repr_shows_at_time(self):
        """Test __repr__ shows at time for daily jobs."""
        sched = schedule.Scheduler()

        def job():
            pass

        j = sched.every().day.at("10:30").do(job)
        job_repr = repr(j)

        assert "10:30" in job_repr
