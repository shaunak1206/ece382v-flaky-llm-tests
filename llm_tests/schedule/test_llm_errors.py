"""
Tests for schedule's exception hierarchy (ScheduleError, ScheduleValueError,
IntervalError) and the situations in the public API that raise them.
"""
import datetime

import pytest

import schedule
from schedule import CancelJob, IntervalError, ScheduleError, ScheduleValueError


def noop():
    pass


# ---------------------------------------------------------------------------
# exception hierarchy
# ---------------------------------------------------------------------------

def test_exception_hierarchy():
    assert issubclass(ScheduleError, Exception)
    assert issubclass(ScheduleValueError, ScheduleError)
    assert issubclass(IntervalError, ScheduleValueError)
    assert issubclass(IntervalError, ScheduleError)
    # CancelJob is a plain sentinel/marker, not an exception.
    assert not issubclass(CancelJob, Exception)


def test_interval_error_can_be_caught_as_schedule_error_or_value_error(scheduler):
    with pytest.raises(ScheduleError):
        scheduler.every(2).second
    with pytest.raises(ScheduleValueError):
        scheduler.every(2).minute


# ---------------------------------------------------------------------------
# IntervalError sources (singular unit + interval != 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("singular", ["second", "minute", "hour", "day", "week"])
def test_interval_error_from_singular_unit_with_bad_interval(scheduler, singular):
    with pytest.raises(IntervalError):
        getattr(scheduler.every(3), singular)


@pytest.mark.parametrize(
    "weekday",
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)
def test_interval_error_from_weekday_with_bad_interval(scheduler, weekday):
    with pytest.raises(IntervalError):
        getattr(scheduler.every(2), weekday)


def test_interval_error_message_mentions_plural_form(scheduler):
    with pytest.raises(IntervalError, match="seconds"):
        scheduler.every(2).second


# ---------------------------------------------------------------------------
# ScheduleValueError sources
# ---------------------------------------------------------------------------

def test_value_error_from_bad_at_time_format(scheduler):
    with pytest.raises(ScheduleValueError, match="daily job"):
        scheduler.every().day.at("not-a-time")


def test_value_error_from_at_on_unsupported_unit(scheduler):
    with pytest.raises(ScheduleValueError):
        scheduler.every().weeks.at("10:30")


def test_value_error_from_do_with_no_unit_set(scheduler):
    with pytest.raises(ScheduleValueError):
        scheduler.every().do(noop)


def test_value_error_from_until_bad_string_or_past_time(scheduler):
    with pytest.raises(ScheduleValueError):
        scheduler.every().seconds.until("this is not a date")
    with pytest.raises(ScheduleValueError):
        scheduler.every().seconds.until(
            datetime.datetime.now() - datetime.timedelta(minutes=5)
        )


def test_value_error_from_invalid_start_day():
    from schedule import _weekday_index

    with pytest.raises(ScheduleValueError):
        _weekday_index("someday")


# ---------------------------------------------------------------------------
# ScheduleError (not a plain ValueError) sources
# ---------------------------------------------------------------------------

def test_schedule_error_from_do_with_no_scheduler():
    from schedule import Job

    job = Job(1, scheduler=None)
    job.unit = "seconds"
    with pytest.raises(ScheduleError):
        job.do(noop)


def test_schedule_error_when_latest_less_than_interval(scheduler):
    job = scheduler.every(10).to(2).seconds
    with pytest.raises(ScheduleError):
        job.do(noop)


# ---------------------------------------------------------------------------
# TypeError sources (deliberately not ScheduleError subclasses)
# ---------------------------------------------------------------------------

def test_type_error_from_at_non_string_argument(scheduler):
    with pytest.raises(TypeError) as excinfo:
        scheduler.every().day.at(1030)
    assert not isinstance(excinfo.value, ScheduleError)


def test_type_error_from_tag_with_unhashable_argument(scheduler):
    job = scheduler.every().seconds.do(noop)
    with pytest.raises(TypeError):
        job.tag({"not": "hashable"})


def test_type_error_from_until_with_unsupported_type(scheduler):
    with pytest.raises(TypeError):
        scheduler.every().seconds.until(3.14)
