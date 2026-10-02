"""Job configuration tests for schedule library (.at, .to, .until, .tag)."""

import pytest
import datetime
import schedule


@pytest.fixture
def clean_scheduler():
    """Provide a clean scheduler for each test."""
    sched = schedule.Scheduler()
    yield sched
    sched.clear()


class TestAtConfiguration:
    """Test .at() method for scheduling at specific times."""

    def test_at_daily_hhmm_format(self, clean_scheduler):
        """Test .at() with daily job using HH:MM format."""
        def job():
            pass

        scheduled = clean_scheduler.every().day.at("10:30").do(job)
        assert scheduled.at_time is not None
        assert scheduled.at_time.hour == 10
        assert scheduled.at_time.minute == 30

    def test_at_daily_hhmmss_format(self, clean_scheduler):
        """Test .at() with daily job using HH:MM:SS format."""
        def job():
            pass

        scheduled = clean_scheduler.every().day.at("14:45:30").do(job)
        assert scheduled.at_time.hour == 14
        assert scheduled.at_time.minute == 45
        assert scheduled.at_time.second == 30

    def test_at_hourly_mmss_format(self, clean_scheduler):
        """Test .at() with hourly job using MM:SS format."""
        def job():
            pass

        scheduled = clean_scheduler.every().hour.at("30:45").do(job)
        assert scheduled.at_time is not None
        assert scheduled.at_time.minute == 30
        assert scheduled.at_time.second == 45

    def test_at_hourly_colon_ss_format(self, clean_scheduler):
        """Test .at() with hourly job using :SS format."""
        def job():
            pass

        scheduled = clean_scheduler.every().hour.at(":45").do(job)
        assert scheduled.at_time is not None
        assert scheduled.at_time.second == 45

    def test_at_minutely_ss_format(self, clean_scheduler):
        """Test .at() with minutely job using :SS format."""
        def job():
            pass

        scheduled = clean_scheduler.every().minute.at(":30").do(job)
        assert scheduled.at_time is not None
        assert scheduled.at_time.second == 30

    def test_at_invalid_format_daily(self, clean_scheduler):
        """Test .at() with invalid time format for daily job."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().day.at("25:00").do(lambda: None)

    def test_at_invalid_format_hourly(self, clean_scheduler):
        """Test .at() with invalid time format for hourly job."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().hour.at("99:99").do(lambda: None)

    def test_at_invalid_format_minutely(self, clean_scheduler):
        """Test .at() with invalid time format for minutely job."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().minute.at("99:99").do(lambda: None)

    def test_at_non_string_raises_typeerror(self, clean_scheduler):
        """Test .at() with non-string raises TypeError."""
        with pytest.raises(TypeError):
            clean_scheduler.every().day.at(1030).do(lambda: None)

    def test_at_with_weekday(self, clean_scheduler):
        """Test .at() with weekday scheduling."""
        def job():
            pass

        scheduled = clean_scheduler.every().monday.at("09:00").do(job)
        assert scheduled.at_time.hour == 9
        assert scheduled.start_day == "monday"

    def test_at_invalid_unit_raises_error(self, clean_scheduler):
        """Test .at() with invalid unit raises ScheduleValueError."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().seconds.at("10:30").do(lambda: None)

    def test_at_hour_boundary_checks(self, clean_scheduler):
        """Test .at() hour boundary validation."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().day.at("24:00").do(lambda: None)

    def test_at_minute_boundary_checks(self, clean_scheduler):
        """Test .at() minute boundary validation."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().day.at("12:60").do(lambda: None)

    def test_at_second_boundary_checks(self, clean_scheduler):
        """Test .at() second boundary validation."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().day.at("12:30:60").do(lambda: None)


class TestToConfiguration:
    """Test .to() method for randomized intervals."""

    def test_to_creates_range(self, clean_scheduler):
        """Test .to() creates a range for randomized intervals."""
        def job():
            pass

        scheduled = clean_scheduler.every(5).to(10).seconds.do(job)
        assert scheduled.latest == 10
        assert scheduled.interval == 5

    def test_to_interval_validation(self, clean_scheduler):
        """Test that latest must be >= interval."""
        # This should fail during _schedule_next_run when latest < interval
        with pytest.raises(schedule.ScheduleError):
            clean_scheduler.every(10).to(5).seconds.do(lambda: None)

    def test_to_with_minutes(self, clean_scheduler):
        """Test .to() with minutes interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(1).to(5).minutes.do(job)
        assert scheduled.latest == 5
        assert scheduled.interval == 1

    def test_to_with_hours(self, clean_scheduler):
        """Test .to() with hours interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(2).to(8).hours.do(job)
        assert scheduled.latest == 8

    def test_to_with_days(self, clean_scheduler):
        """Test .to() with days interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(1).to(7).days.do(job)
        assert scheduled.latest == 7

    def test_to_with_weeks(self, clean_scheduler):
        """Test .to() with weeks interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(1).to(4).weeks.do(job)
        assert scheduled.latest == 4

    def test_to_equal_to_interval(self, clean_scheduler):
        """Test .to() when latest equals interval."""
        def job():
            pass

        scheduled = clean_scheduler.every(5).to(5).seconds.do(job)
        assert scheduled.latest == 5


class TestUntilConfiguration:
    """Test .until() method for job deadline."""

    def test_until_with_datetime(self, clean_scheduler):
        """Test .until() with datetime object."""
        def job():
            pass

        future_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        scheduled = clean_scheduler.every().seconds.until(future_time).do(job)
        assert scheduled.cancel_after is not None
        assert scheduled.cancel_after == future_time

    def test_until_with_timedelta(self, clean_scheduler):
        """Test .until() with timedelta object."""
        def job():
            pass

        delta = datetime.timedelta(hours=2)
        scheduled = clean_scheduler.every().seconds.until(delta).do(job)
        assert scheduled.cancel_after is not None
        # Should be approximately 2 hours from now
        assert (scheduled.cancel_after - datetime.datetime.now()).total_seconds() < 7210  # 2 hours + 10 sec tolerance

    def test_until_with_time_object(self, clean_scheduler):
        """Test .until() with time object (today at that time)."""
        def job():
            pass

        future_time = (datetime.datetime.now() + datetime.timedelta(hours=1)).time()
        scheduled = clean_scheduler.every().seconds.until(future_time).do(job)
        assert scheduled.cancel_after is not None

    def test_until_with_string_datetime_format(self, clean_scheduler):
        """Test .until() with string in datetime format."""
        def job():
            pass

        future = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        scheduled = clean_scheduler.every().seconds.until(future).do(job)
        assert scheduled.cancel_after is not None

    def test_until_with_string_date_format(self, clean_scheduler):
        """Test .until() with string in date-only format."""
        def job():
            pass

        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        scheduled = clean_scheduler.every().seconds.until(tomorrow).do(job)
        assert scheduled.cancel_after is not None

    def test_until_with_string_time_format(self, clean_scheduler):
        """Test .until() with string in time-only format."""
        def job():
            pass

        future_time = (datetime.datetime.now() + datetime.timedelta(hours=2)).strftime("%H:%M:%S")
        scheduled = clean_scheduler.every().seconds.until(future_time).do(job)
        assert scheduled.cancel_after is not None

    def test_until_past_time_raises_error(self, clean_scheduler):
        """Test .until() with past datetime raises ScheduleValueError."""
        past_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().seconds.until(past_time).do(lambda: None)

    def test_until_invalid_string_format_raises_error(self, clean_scheduler):
        """Test .until() with invalid string format raises ScheduleValueError."""
        with pytest.raises(schedule.ScheduleValueError):
            clean_scheduler.every().seconds.until("not-a-date").do(lambda: None)

    def test_until_invalid_type_raises_error(self, clean_scheduler):
        """Test .until() with invalid type raises TypeError."""
        with pytest.raises(TypeError):
            clean_scheduler.every().seconds.until(12345).do(lambda: None)


class TestTagConfiguration:
    """Test .tag() method for labeling jobs."""

    def test_tag_single_tag(self, clean_scheduler):
        """Test tagging a job with a single tag."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag("my-tag")
        assert "my-tag" in scheduled.tags

    def test_tag_multiple_tags(self, clean_scheduler):
        """Test tagging a job with multiple tags."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag("tag1", "tag2", "tag3")
        assert "tag1" in scheduled.tags
        assert "tag2" in scheduled.tags
        assert "tag3" in scheduled.tags
        assert len(scheduled.tags) == 3

    def test_tag_chain_calls(self, clean_scheduler):
        """Test chaining multiple .tag() calls."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag("first").tag("second")
        assert "first" in scheduled.tags
        assert "second" in scheduled.tags

    def test_tag_non_hashable_raises_error(self, clean_scheduler):
        """Test that non-hashable tags raise TypeError."""
        with pytest.raises(TypeError):
            clean_scheduler.every().seconds.do(lambda: None).tag(["list"])

    def test_tag_returns_job(self, clean_scheduler):
        """Test that .tag() returns the job for chaining."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job)
        result = scheduled.tag("test")
        assert result is scheduled

    def test_tag_with_different_types(self, clean_scheduler):
        """Test tagging with different hashable types."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag(1, "string", ("tuple",), 2.5)
        assert 1 in scheduled.tags
        assert "string" in scheduled.tags
        assert ("tuple",) in scheduled.tags
        assert 2.5 in scheduled.tags

    def test_tag_duplicate_tags_deduplicated(self, clean_scheduler):
        """Test that duplicate tags are deduplicated (set behavior)."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag("dup", "dup", "dup")
        assert len(scheduled.tags) == 1

    def test_tag_with_none_is_hashable(self, clean_scheduler):
        """Test that None can be used as a tag."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag(None)
        assert None in scheduled.tags


class TestMethodChaining:
    """Test that configuration methods can be chained."""

    def test_chain_at_and_do(self, clean_scheduler):
        """Test chaining .at() and .do()."""
        def job():
            return "result"

        scheduled = clean_scheduler.every().day.at("10:00").do(job)
        assert scheduled.at_time is not None
        assert scheduled.job_func is not None

    def test_chain_to_and_do(self, clean_scheduler):
        """Test chaining .to() and .do()."""
        def job():
            pass

        scheduled = clean_scheduler.every(1).to(5).seconds.do(job)
        assert scheduled.latest == 5
        assert scheduled.job_func is not None

    def test_chain_to_at_and_do(self, clean_scheduler):
        """Test chaining .to(), .at(), and .do()."""
        def job():
            pass

        scheduled = clean_scheduler.every(1).to(5).days.at("10:00").do(job)
        assert scheduled.latest == 5
        assert scheduled.at_time is not None

    def test_chain_tag_do_tag(self, clean_scheduler):
        """Test chaining .tag(), .do(), .tag()."""
        def job():
            pass

        scheduled = clean_scheduler.every().seconds.do(job).tag("first").tag("second")
        assert "first" in scheduled.tags
        assert "second" in scheduled.tags

    def test_chain_until_and_do(self, clean_scheduler):
        """Test chaining .until() and .do()."""
        def job():
            pass

        future = datetime.datetime.now() + datetime.timedelta(hours=1)
        scheduled = clean_scheduler.every().seconds.until(future).do(job)
        assert scheduled.cancel_after is not None
        assert scheduled.job_func is not None

    def test_full_chain(self, clean_scheduler):
        """Test comprehensive method chaining."""
        def job():
            return 42

        future = datetime.datetime.now() + datetime.timedelta(hours=1)
        scheduled = (
            clean_scheduler.every(1).to(5).days
            .at("10:00")
            .until(future)
            .do(job)
            .tag("important")
            .tag("daily")
        )
        assert scheduled.latest == 5
        assert scheduled.at_time is not None
        assert scheduled.cancel_after is not None
        assert "important" in scheduled.tags
        assert "daily" in scheduled.tags
