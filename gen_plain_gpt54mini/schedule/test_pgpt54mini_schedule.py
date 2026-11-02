import datetime as dt
from collections.abc import Hashable

import pytest
from freezegun import freeze_time

import schedule


@pytest.fixture(autouse=True)
def clear_default_scheduler():
    schedule.clear()
    yield
    schedule.clear()


def make_job(interval=1):
    return schedule.Scheduler().every(interval)


def test_scheduler_starts_empty():
    s = schedule.Scheduler()
    assert s.jobs == []


def test_every_returns_job_with_interval():
    s = schedule.Scheduler()
    job = s.every(3)
    assert isinstance(job, schedule.Job)
    assert job.interval == 3
    assert job.scheduler is s


def test_default_every_uses_default_scheduler():
    job = schedule.every(2)
    assert job.scheduler is schedule.default_scheduler


def test_job_second_sets_unit_seconds():
    job = schedule.Scheduler().every(1).second
    assert job.unit == "seconds"


def test_job_second_raises_for_interval_not_one():
    job = schedule.Scheduler().every(2)
    with pytest.raises(schedule.IntervalError):
        _ = job.second


def test_job_minute_sets_unit_minutes():
    job = schedule.Scheduler().every(1).minute
    assert job.unit == "minutes"


def test_job_minute_raises_for_interval_not_one():
    job = schedule.Scheduler().every(2)
    with pytest.raises(schedule.IntervalError):
        _ = job.minute


def test_job_hour_sets_unit_hours():
    job = schedule.Scheduler().every(1).hour
    assert job.unit == "hours"


def test_job_day_sets_unit_days():
    job = schedule.Scheduler().every(1).day
    assert job.unit == "days"


def test_job_week_sets_unit_weeks():
    job = schedule.Scheduler().every(1).week
    assert job.unit == "weeks"


def test_job_monday_sets_start_day_and_weeks_unit():
    job = schedule.Scheduler().every(1).monday
    assert job.start_day == "monday"
    assert job.unit == "weeks"


def test_job_monday_raises_for_interval_gt_one():
    job = schedule.Scheduler().every(2)
    with pytest.raises(schedule.IntervalError):
        _ = job.monday


def test_job_tuesday_sets_start_day():
    job = schedule.Scheduler().every(1).tuesday
    assert job.start_day == "tuesday"


def test_job_wednesday_sets_start_day():
    job = schedule.Scheduler().every(1).wednesday
    assert job.start_day == "wednesday"


def test_job_thursday_sets_start_day():
    job = schedule.Scheduler().every(1).thursday
    assert job.start_day == "thursday"


def test_job_friday_sets_start_day():
    job = schedule.Scheduler().every(1).friday
    assert job.start_day == "friday"


def test_job_saturday_sets_start_day():
    job = schedule.Scheduler().every(1).saturday
    assert job.start_day == "saturday"


def test_job_sunday_sets_start_day():
    job = schedule.Scheduler().every(1).sunday
    assert job.start_day == "sunday"


def test_tag_adds_unique_hashable_tags():
    job = schedule.Scheduler().every(1)
    obj = object()
    job.tag("x", "x", 1, obj)
    assert job.tags == {"x", 1, obj}


def test_tag_returns_self():
    job = schedule.Scheduler().every(1)
    assert job.tag("x") is job


def test_tag_rejects_unhashable():
    job = schedule.Scheduler().every(1)
    with pytest.raises(TypeError):
        job.tag([])


def test_at_rejects_invalid_unit():
    job = schedule.Scheduler().every(1)
    with pytest.raises(schedule.ScheduleValueError):
        job.at("10:00")


def test_at_rejects_non_string():
    job = schedule.Scheduler().every(1).day
    with pytest.raises(TypeError):
        job.at(123)  # type: ignore[arg-type]


def test_at_daily_accepts_hh_mm():
    job = schedule.Scheduler().every(1).day.at("10:30")
    assert job.at_time == dt.time(10, 30)


def test_at_daily_accepts_hh_mm_ss():
    job = schedule.Scheduler().every(1).day.at("10:30:15")
    assert job.at_time == dt.time(10, 30, 15)


def test_at_daily_rejects_bad_format():
    job = schedule.Scheduler().every(1).day
    with pytest.raises(schedule.ScheduleValueError):
        job.at("99:99")


def test_at_daily_rejects_out_of_range_hour():
    job = schedule.Scheduler().every(1).day
    with pytest.raises(schedule.ScheduleValueError):
        job.at("24:00")


def test_at_hourly_accepts_colon_format():
    job = schedule.Scheduler().every(1).hour.at(":30")
    assert job.at_time == dt.time(0, 30)


def test_at_hourly_accepts_mm_ss_format():
    job = schedule.Scheduler().every(1).hour.at("05:30")
    assert job.at_time == dt.time(0, 5, 30)


def test_at_hourly_rejects_bad_format():
    job = schedule.Scheduler().every(1).hour
    with pytest.raises(schedule.ScheduleValueError):
        job.at("1:2:3")


def test_at_minutely_accepts_ss_format():
    job = schedule.Scheduler().every(1).minute.at(":45")
    assert job.at_time == dt.time(0, 0, 45)


def test_at_minutely_rejects_bad_format():
    job = schedule.Scheduler().every(1).minute
    with pytest.raises(schedule.ScheduleValueError):
        job.at("10:10")


def test_at_timezone_invalid_type_rejected():
    job = schedule.Scheduler().every(1).day
    with pytest.raises(schedule.ScheduleValueError):
        job.at("10:00", tz=object())  # type: ignore[arg-type]


def test_to_sets_latest():
    job = schedule.Scheduler().every(1).seconds.to(5)
    assert job.latest == 5


def test_to_returns_self():
    job = schedule.Scheduler().every(1).seconds
    assert job.to(5) is job


def test_until_rejects_invalid_type():
    job = schedule.Scheduler().every(1).seconds
    with pytest.raises(TypeError):
        job.until(123)  # type: ignore[arg-type]


def test_until_accepts_datetime():
    when = dt.datetime(2025, 1, 1, 12, 0, 0)
    job = schedule.Scheduler().every(1).seconds.until(when)
    assert job.cancel_after == when


def test_until_accepts_timedelta():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds.until(dt.timedelta(seconds=10))
        assert job.cancel_after == dt.datetime(2025, 1, 1, 12, 0, 10)


def test_until_accepts_time():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds.until(dt.time(12, 0, 10))
        assert job.cancel_after == dt.datetime(2025, 1, 1, 12, 0, 10)


def test_until_accepts_date_string():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds.until("2025-01-02")
        assert job.cancel_after == dt.datetime(2025, 1, 2, 0, 0, 0)


def test_until_accepts_time_string():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds.until("12:00:10")
        assert job.cancel_after == dt.datetime(2025, 1, 1, 12, 0, 10)


def test_until_rejects_invalid_string():
    job = schedule.Scheduler().every(1).seconds
    with pytest.raises(schedule.ScheduleValueError):
        job.until("not-a-date")


def test_until_rejects_past_time():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds
        with pytest.raises(schedule.ScheduleValueError):
            job.until("2025-01-01 11:59:59")


def test_do_registers_job_and_wraps_function():
    s = schedule.Scheduler()

    def f(x):
        return x

    job = s.every(1).seconds.do(f, 5)
    assert job in s.jobs
    assert job.job_func() == 5
    assert job.job_func.__name__ == "f"


def test_do_raises_without_scheduler():
    job = schedule.Job(1, scheduler=None)
    job.seconds

    def f():
        return None

    with pytest.raises(schedule.ScheduleError):
        job.do(f)


def test_should_run_uses_next_run():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds
        job.next_run = dt.datetime(2025, 1, 1, 11, 59, 59)
        assert job.should_run is True


def test_should_run_false_when_future():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.Scheduler().every(1).seconds
        job.next_run = dt.datetime(2025, 1, 1, 12, 0, 1)
        assert job.should_run is False


def test_run_returns_job_result():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()

        def f():
            return "ok"

        job = s.every(1).seconds.do(f)
        job.cancel_after = dt.datetime(2025, 1, 1, 12, 1, 0)
        assert job.run() == "ok"


def test_run_updates_last_run():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()

        def f():
            return "ok"

        job = s.every(1).seconds.do(f)
        job.cancel_after = dt.datetime(2025, 1, 1, 12, 1, 0)
        job.run()
        assert job.last_run == dt.datetime(2025, 1, 1, 12, 0, 0)


def test_run_returns_canceljob_if_overdue_before_run():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()
        called = []

        def f():
            called.append(True)

        job = s.every(1).seconds.do(f)
        job.cancel_after = dt.datetime(2025, 1, 1, 11, 59, 59)
        assert job.run() is schedule.CancelJob
        assert called == []


def test_run_returns_canceljob_if_next_run_overdue():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()

        def f():
            return "ok"

        job = s.every(1).seconds.do(f)
        job.cancel_after = dt.datetime(2025, 1, 1, 12, 0, 0)
        assert job.run() is schedule.CancelJob


def test_scheduler_run_pending_runs_due_jobs_only():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()
        called = []

        def a():
            called.append("a")

        def b():
            called.append("b")

        j1 = s.every(1).seconds.do(a)
        j2 = s.every(1).seconds.do(b)
        j1.next_run = dt.datetime(2025, 1, 1, 11, 59, 59)
        j2.next_run = dt.datetime(2025, 1, 1, 12, 0, 1)
        s.run_pending()
        assert called == ["a"]


def test_scheduler_run_all_runs_all_jobs():
    s = schedule.Scheduler()
    called = []

    def f():
        called.append(True)

    s.every(1).seconds.do(f)
    s.every(1).seconds.do(f)
    s.run_all()
    assert len(called) == 2


def test_scheduler_run_all_sleeps_between_jobs(monkeypatch):
    s = schedule.Scheduler()
    calls = []

    def f():
        calls.append("x")

    s.every(1).seconds.do(f)
    s.every(1).seconds.do(f)
    monkeypatch.setattr(schedule.time, "sleep", lambda x: calls.append(("sleep", x)))
    s.run_all(delay_seconds=3)
    assert calls == ["x", ("sleep", 3), "x", ("sleep", 3)]


def test_get_jobs_returns_copy():
    s = schedule.Scheduler()
    job = s.every(1).seconds.do(lambda: None)
    jobs = s.get_jobs()
    jobs.clear()
    assert s.jobs == [job]


def test_get_jobs_filters_by_tag():
    s = schedule.Scheduler()
    j1 = s.every(1).seconds.do(lambda: None).tag("x")
    j2 = s.every(1).seconds.do(lambda: None).tag("y")
    assert s.get_jobs("x") == [j1]
    assert s.get_jobs("y") == [j2]


def test_clear_removes_all_jobs():
    s = schedule.Scheduler()
    s.every(1).seconds.do(lambda: None)
    s.clear()
    assert s.jobs == []


def test_clear_by_tag_removes_only_tagged_jobs():
    s = schedule.Scheduler()
    j1 = s.every(1).seconds.do(lambda: None).tag("x")
    j2 = s.every(1).seconds.do(lambda: None).tag("y")
    s.clear("x")
    assert s.jobs == [j2]
    assert j1 not in s.jobs


def test_cancel_job_removes_job():
    s = schedule.Scheduler()
    job = s.every(1).seconds.do(lambda: None)
    s.cancel_job(job)
    assert job not in s.jobs


def test_cancel_job_ignores_missing_job():
    s = schedule.Scheduler()
    job = schedule.Job(1, s)
    s.cancel_job(job)
    assert s.jobs == []


def test_get_next_run_none_when_no_jobs():
    s = schedule.Scheduler()
    assert s.get_next_run() is None


def test_get_next_run_returns_earliest():
    s = schedule.Scheduler()
    j1 = s.every(1).seconds.do(lambda: None)
    j2 = s.every(1).seconds.do(lambda: None)
    j1.next_run = dt.datetime(2025, 1, 1, 12, 0, 10)
    j2.next_run = dt.datetime(2025, 1, 1, 12, 0, 5)
    assert s.get_next_run() == j2.next_run


def test_idle_seconds_none_when_no_jobs():
    s = schedule.Scheduler()
    assert s.idle_seconds is None


def test_idle_seconds_computes_difference():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()
        j = s.every(1).seconds.do(lambda: None)
        j.next_run = dt.datetime(2025, 1, 1, 12, 0, 10)
        assert s.idle_seconds == 10.0


def test_repeat_decorator_schedules_function():
    s = schedule.Scheduler()

    @schedule.repeat(s.every(1).seconds)
    def f():
        return "ok"

    assert len(s.jobs) == 1
    assert s.jobs[0].job_func() == "ok"
    assert f() == "ok"


def test_move_to_next_weekday_same_day_no_move():
    moment = dt.datetime(2025, 1, 6, 10, 0, 0)  # Monday
    assert schedule._move_to_next_weekday(moment, "monday") == moment


def test_move_to_next_weekday_later_same_week():
    moment = dt.datetime(2025, 1, 6, 10, 0, 0)  # Monday
    assert schedule._move_to_next_weekday(moment, "wednesday").weekday() == 2


def test_move_to_next_weekday_wraps_to_next_week():
    moment = dt.datetime(2025, 1, 10, 10, 0, 0)  # Friday
    assert schedule._move_to_next_weekday(moment, "monday").weekday() == 0


def test_weekday_index_valid():
    assert schedule._weekday_index("monday") == 0
    assert schedule._weekday_index("sunday") == 6


def test_weekday_index_invalid_raises():
    with pytest.raises(schedule.ScheduleValueError):
        schedule._weekday_index("funday")


def test_job_str_contains_fields():
    s = schedule.Scheduler()

    def f():
        return None

    job = s.every(2).hours.do(f)
    text = str(job)
    assert "interval=2" in text
    assert "unit=hours" in text
    assert "do=f" in text


def test_job_repr_without_job_func():
    job = schedule.Job(1)
    text = repr(job)
    assert "[None]" in text
    assert "Every 1" in text


def test_job_repr_with_args_and_kwargs():
    s = schedule.Scheduler()

    def f(a, b=2):
        return a + b

    job = s.every(1).seconds.do(f, 1, b=3)
    text = repr(job)
    assert "f(1, b=3)" in text


def test_job_comparison_uses_next_run():
    j1 = schedule.Job(1)
    j2 = schedule.Job(1)
    j1.next_run = dt.datetime(2025, 1, 1, 12, 0, 1)
    j2.next_run = dt.datetime(2025, 1, 1, 12, 0, 2)
    assert j1 < j2


def test_default_clear_removes_default_jobs():
    schedule.every(1).seconds.do(lambda: None)
    assert schedule.get_jobs()
    schedule.clear()
    assert schedule.get_jobs() == []


def test_default_cancel_job_removes_from_default_scheduler():
    job = schedule.every(1).seconds.do(lambda: None)
    schedule.cancel_job(job)
    assert job not in schedule.get_jobs()


def test_default_next_run_property():
    job = schedule.every(1).seconds.do(lambda: None)
    assert schedule.next_run() == job.next_run


def test_default_idle_seconds_property():
    with freeze_time("2025-01-01 12:00:00"):
        job = schedule.every(1).seconds.do(lambda: None)
        job.next_run = dt.datetime(2025, 1, 1, 12, 0, 5)
        assert schedule.idle_seconds() == 5.0


def test_scheduler_run_pending_skips_when_no_due_jobs():
    with freeze_time("2025-01-01 12:00:00"):
        s = schedule.Scheduler()
        called = []
        j = s.every(1).seconds.do(lambda: called.append(True))
        j.next_run = dt.datetime(2025, 1, 1, 12, 0, 1)
        s.run_pending()
        assert called == []


def test_run_all_uses_copy_of_jobs_list():
    s = schedule.Scheduler()
    called = []

    def f():
        called.append(True)
        s.clear()

    s.every(1).seconds.do(f)
    s.every(1).seconds.do(lambda: called.append(False))
    s.run_all()
    assert called[0] is True


def test_latest_less_than_interval_raises_on_schedule_next_run():
    s = schedule.Scheduler()
    job = s.every(5).seconds.to(4)
    job.job_func = lambda: None
    with pytest.raises(schedule.ScheduleError):
        job._schedule_next_run()


def test_invalid_unit_raises_on_schedule_next_run():
    job = schedule.Job(1)
    with pytest.raises(schedule.ScheduleValueError):
        job._schedule_next_run()


def test_is_overdue_false_without_cancel_after():
    job = schedule.Job(1)
    assert job._is_overdue(dt.datetime.now()) is False


def test_is_overdue_true_after_deadline():
    job = schedule.Job(1)
    job.cancel_after = dt.datetime(2025, 1, 1, 0, 0, 0)
    assert job._is_overdue(dt.datetime(2025, 1, 1, 0, 0, 1)) is True


def test_decode_datetimestr_returns_none_for_invalid():
    job = schedule.Job(1)
    assert job._decode_datetimestr("x", ["%Y-%m-%d"]) is None


def test_decode_datetimestr_parses_valid():
    job = schedule.Job(1)
    parsed = job._decode_datetimestr("2025-01-02", ["%Y-%m-%d"])
    assert parsed == dt.datetime(2025, 1, 2, 0, 0, 0)
