import threading
from datetime import timedelta

import pytest

from tenacity import (
    RetryCallState,
    Retrying,
    stop_after_attempt,
    stop_after_delay,
    stop_all,
    stop_any,
    stop_before_delay,
    stop_never,
    stop_when_event_set,
)


def make_state(attempt_number=1, elapsed=None, upcoming_sleep=0.0):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    rs.attempt_number = attempt_number
    if elapsed is not None:
        rs.start_time = 100.0
        rs.outcome_timestamp = 100.0 + elapsed
    rs.upcoming_sleep = upcoming_sleep
    return rs


def test_stop_after_attempt_boundary_and_stop_never():
    s = stop_after_attempt(3)
    assert s(make_state(1)) is False
    assert s(make_state(2)) is False
    assert s(make_state(3)) is True
    assert s(make_state(4)) is True
    assert stop_after_attempt(1)(make_state(1)) is True
    assert stop_never(make_state(10**6)) is False


def test_stop_after_delay_compares_elapsed_time_to_limit():
    s = stop_after_delay(5)
    assert s(make_state(elapsed=4.5)) is False
    assert s(make_state(elapsed=5)) is True
    assert s(make_state(elapsed=6)) is True
    assert stop_after_delay(timedelta(seconds=2))(make_state(elapsed=2.5)) is True


def test_stop_after_delay_requires_outcome_timestamp():
    with pytest.raises(RuntimeError):
        stop_after_delay(5)(make_state(elapsed=None))


def test_stop_before_delay_accounts_for_upcoming_sleep():
    s = stop_before_delay(10)
    assert s(make_state(elapsed=3, upcoming_sleep=2)) is False
    assert s(make_state(elapsed=3, upcoming_sleep=7)) is True
    assert s(make_state(elapsed=3, upcoming_sleep=8)) is True
    with pytest.raises(RuntimeError):
        s(make_state(elapsed=None))


def test_stop_when_event_set_follows_event_state():
    event = threading.Event()
    s = stop_when_event_set(event)
    assert s(make_state()) is False
    event.set()
    assert s(make_state()) is True
    event.clear()
    assert s(make_state()) is False


def test_stop_any_all_and_operators():
    a2 = stop_after_attempt(2)
    a4 = stop_after_attempt(4)
    state3 = make_state(3)
    assert stop_any(a2, a4)(state3) is True
    assert stop_all(a2, a4)(state3) is False
    assert stop_all(a2, a4)(make_state(4)) is True
    assert isinstance(a2 | a4, stop_any)
    assert isinstance(a2 & a4, stop_all)
    assert (a2 | a4)(state3) is True
    assert (a2 & a4)(state3) is False
    # degenerate cases mirror any()/all() on empty input
    assert stop_any()(state3) is False
    assert stop_all()(state3) is True
