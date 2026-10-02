"""Basic scheduling tests for schedule library."""

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


class TestBasicIntervals:
    """Test basic job scheduling with different intervals."""

    def test_schedule_every_seconds(self, clean_scheduler):
        """Test scheduling a job to run every N seconds."""
        call_count = []

        def job():
            call_count.append(1)

        scheduled = clean_scheduler.every(5).seconds.do(job)
        assert scheduled is not None
        assert scheduled.unit == "seconds"
        assert scheduled.interval == 5
        assert len(clean_scheduler.jobs) == 1

    def test_schedule_every_minutes(self, clean_scheduler):
        """Test scheduling a job to run every N minutes."""
        def job():
            pass

        scheduled = clean_scheduler.every(10).minutes.do(job)
        assert scheduled.unit == "minutes"
        assert scheduled.interval == 10

    def test_schedule_every_hours(self, clean_scheduler):
        """Test scheduling a job to run every N hours."""
        def job():
            pass

        scheduled = clean_scheduler.every(2).hours.do(job)
        assert scheduled.unit == "hours"
        assert scheduled.interval == 2

    def test_schedule_every_days(self, clean_scheduler):
        """Test scheduling a job to run every N days."""
        def job():
            pass

        scheduled = clean_scheduler.every(3).days.do(job)
        assert scheduled.unit == "days"
        assert scheduled.interval == 3

    def test_schedule_every_weeks(self, clean_scheduler):
        """Test scheduling a job to run every N weeks."""
        def job():
            pass

        scheduled = clean_scheduler.every(2).weeks.do(job)
        assert scheduled.unit == "weeks"
        assert scheduled.interval == 2

    def test_schedule_every_default_interval(self, clean_scheduler):
        """Test that default interval is 1."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.interval == 1


class TestSingleUnitSyntax:
    """Test singular time unit syntax."""

    def test_every_second(self, clean_scheduler):
        """Test every().second syntax (singular)."""
        def job():
            pass

        scheduled = clean_scheduler.every().second.do(job)
        assert scheduled.unit == "seconds"

    def test_every_minute(self, clean_scheduler):
        """Test every().minute syntax (singular)."""
        def job():
            pass

        scheduled = clean_scheduler.every().minute.do(job)
        assert scheduled.unit == "minutes"

    def test_every_hour(self, clean_scheduler):
        """Test every().hour syntax (singular)."""
        def job():
            pass

        scheduled = clean_scheduler.every().hour.do(job)
        assert scheduled.unit == "hours"

    def test_every_day(self, clean_scheduler):
        """Test every().day syntax (singular)."""
        def job():
            pass

        scheduled = clean_scheduler.every().day.do(job)
        assert scheduled.unit == "days"

    def test_every_week(self, clean_scheduler):
        """Test every().week syntax (singular)."""
        def job():
            pass

        scheduled = clean_scheduler.every().week.do(job)
        assert scheduled.unit == "weeks"

    def test_singular_with_multiple_interval_raises_error(self, clean_scheduler):
        """Test that using singular form with interval > 1 raises IntervalError."""
        with pytest.raises(schedule.IntervalError):
            clean_scheduler.every(2).second

    def test_singular_minutes_with_interval_raises_error(self, clean_scheduler):
        """Test that using singular minute with interval > 1 raises IntervalError."""
        with pytest.raises(schedule.IntervalError):
            clean_scheduler.every(3).minute


class TestWeekdayScheduling:
    """Test scheduling jobs for specific weekdays."""

    def test_schedule_monday(self, clean_scheduler):
        """Test scheduling a job for Monday."""
        def job():
            pass

        scheduled = clean_scheduler.every().monday.do(job)
        assert scheduled.start_day == "monday"
        assert scheduled.unit == "weeks"

    def test_schedule_tuesday(self, clean_scheduler):
        """Test scheduling a job for Tuesday."""
        def job():
            pass

        scheduled = clean_scheduler.every().tuesday.do(job)
        assert scheduled.start_day == "tuesday"

    def test_schedule_wednesday(self, clean_scheduler):
        """Test scheduling a job for Wednesday."""
        def job():
            pass

        scheduled = clean_scheduler.every().wednesday.do(job)
        assert scheduled.start_day == "wednesday"

    def test_schedule_thursday(self, clean_scheduler):
        """Test scheduling a job for Thursday."""
        def job():
            pass

        scheduled = clean_scheduler.every().thursday.do(job)
        assert scheduled.start_day == "thursday"

    def test_schedule_friday(self, clean_scheduler):
        """Test scheduling a job for Friday."""
        def job():
            pass

        scheduled = clean_scheduler.every().friday.do(job)
        assert scheduled.start_day == "friday"

    def test_schedule_saturday(self, clean_scheduler):
        """Test scheduling a job for Saturday."""
        def job():
            pass

        scheduled = clean_scheduler.every().saturday.do(job)
        assert scheduled.start_day == "saturday"

    def test_schedule_sunday(self, clean_scheduler):
        """Test scheduling a job for Sunday."""
        def job():
            pass

        scheduled = clean_scheduler.every().sunday.do(job)
        assert scheduled.start_day == "sunday"

    def test_weekday_with_interval_raises_error(self, clean_scheduler):
        """Test that weekday scheduling with interval > 1 raises IntervalError."""
        with pytest.raises(schedule.IntervalError):
            clean_scheduler.every(2).monday


class TestJobCreation:
    """Test job creation and configuration."""

    def test_do_with_function(self, clean_scheduler):
        """Test scheduling a function with do()."""
        def my_job():
            return "executed"

        scheduled = clean_scheduler.every().seconds.do(my_job)
        assert scheduled.job_func is not None
        result = scheduled.job_func()
        assert result == "executed"

    def test_do_with_callable_object(self, clean_scheduler):
        """Test scheduling a callable object."""
        class CallableJob:
            def __call__(self):
                return "called"

        job_obj = CallableJob()
        scheduled = clean_scheduler.every().seconds.do(job_obj)
        result = scheduled.job_func()
        assert result == "called"

    def test_do_with_args(self, clean_scheduler):
        """Test scheduling a function with positional arguments."""
        def job_with_args(a, b):
            return a + b

        scheduled = clean_scheduler.every().seconds.do(job_with_args, 2, 3)
        result = scheduled.job_func()
        assert result == 5

    def test_do_with_kwargs(self, clean_scheduler):
        """Test scheduling a function with keyword arguments."""
        def job_with_kwargs(x=0, y=0):
            return x * y

        scheduled = clean_scheduler.every().seconds.do(job_with_kwargs, x=4, y=5)
        result = scheduled.job_func()
        assert result == 20

    def test_do_with_mixed_args(self, clean_scheduler):
        """Test scheduling a function with mixed args and kwargs."""
        def job_mixed(a, b, c=0):
            return a + b + c

        scheduled = clean_scheduler.every().seconds.do(job_mixed, 1, 2, c=3)
        result = scheduled.job_func()
        assert result == 6

    def test_job_registered_in_scheduler(self, clean_scheduler):
        """Test that scheduled job is added to scheduler's jobs list."""
        def job():
            pass

        assert len(clean_scheduler.jobs) == 0
        clean_scheduler.every().seconds.do(job)
        assert len(clean_scheduler.jobs) == 1

    def test_multiple_jobs_registered(self, clean_scheduler):
        """Test scheduling multiple jobs."""
        def job1():
            pass

        def job2():
            pass

        clean_scheduler.every().seconds.do(job1)
        clean_scheduler.every().minutes.do(job2)
        assert len(clean_scheduler.jobs) == 2


class TestJobProperties:
    """Test Job properties and attributes."""

    def test_job_next_run_is_set(self, clean_scheduler):
        """Test that next_run is set when job is scheduled."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.next_run is not None
        assert isinstance(scheduled.next_run, datetime.datetime)

    def test_job_last_run_initially_none(self, clean_scheduler):
        """Test that last_run is None initially."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert scheduled.last_run is None

    def test_job_should_run_property(self, clean_scheduler):
        """Test the should_run property of a job."""
        def job():
            pass

        scheduled = clean_scheduler.every(10).seconds.do(job)
        # Job scheduled 10 seconds from now should not run immediately
        assert not scheduled.should_run

    def test_job_interval_property(self, clean_scheduler):
        """Test accessing job interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(7).minutes.do(job)
        assert scheduled.interval == 7

    def test_job_unit_property(self, clean_scheduler):
        """Test accessing job unit."""
        def job():
            pass

        scheduled = clean_scheduler.every(7).hours.do(job)
        assert scheduled.unit == "hours"

    def test_job_tags_initially_empty(self, clean_scheduler):
        """Test that tags set is initially empty."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        assert len(scheduled.tags) == 0

    def test_job_str_representation(self, clean_scheduler):
        """Test string representation of a job."""
        def my_function():
            pass

        scheduled = clean_scheduler.every(5).minutes.do(my_function)
        job_str = str(scheduled)
        assert "5" in job_str
        assert "minute" in job_str
