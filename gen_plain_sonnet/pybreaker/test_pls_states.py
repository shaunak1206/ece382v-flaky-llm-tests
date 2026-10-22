"""Tests for state transitions of pybreaker circuit breakers."""

import sys
from datetime import datetime, timedelta, timezone

import pytest
from freezegun import freeze_time

try:
    import pybreaker
except ImportError:  # pragma: no cover - fallback for src layout
    sys.path.insert(0, "/Users/shaunkapoor/ece382v-flaky-llm-tests/subjects/pybreaker/src")
    import pybreaker

from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
)

START = "2024-01-01 12:00:00"


class Boom(Exception):
    pass


def fail():
    raise Boom("boom")


def ok():
    return "ok"


def trip(breaker):
    """Drive a breaker with fail_max failing calls until it opens."""
    for _ in range(breaker.fail_max):
        with pytest.raises((Boom, CircuitBreakerError)):
            breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_closed_success_returns_value_and_stays_closed():
    breaker = CircuitBreaker()
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0


def test_closed_failure_below_threshold_reraises_original_exception():
    breaker = CircuitBreaker(fail_max=3)
    for expected in (1, 2):
        with pytest.raises(Boom):
            breaker.call(fail)
        assert breaker.fail_counter == expected
        assert breaker.current_state == STATE_CLOSED


def test_closed_trips_at_fail_max_with_circuit_breaker_error():
    breaker = CircuitBreaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    with pytest.raises(CircuitBreakerError, match="Failures threshold reached"):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN
    assert breaker.fail_counter == 3


def test_open_state_rejects_calls_without_running_function():
    breaker = CircuitBreaker(fail_max=1, reset_timeout=60)
    trip(breaker)
    calls = []

    def tracked():
        calls.append(1)
        return "ran"

    for _ in range(3):
        with pytest.raises(CircuitBreakerError, match="Timeout not elapsed"):
            breaker.call(tracked)
    assert calls == []
    assert breaker.fail_counter == 1
    assert breaker.current_state == STATE_OPEN


def test_success_resets_fail_counter_so_full_threshold_is_needed_again():
    breaker = CircuitBreaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.call(ok) == "ok"
    assert breaker.fail_counter == 0
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.current_state == STATE_CLOSED


def test_fail_max_of_one_trips_on_first_failure():
    breaker = CircuitBreaker(fail_max=1)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_fail_max_setter_affects_existing_breaker():
    breaker = CircuitBreaker(fail_max=5)
    with pytest.raises(Boom):
        breaker.call(fail)
    breaker.fail_max = 2
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_throw_new_error_on_trip_false_reraises_original_when_tripping():
    breaker = CircuitBreaker(fail_max=2, throw_new_error_on_trip=False)
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)


def test_open_method_returns_throw_flag_and_opens_circuit():
    assert CircuitBreaker().open() is True
    breaker = CircuitBreaker(throw_new_error_on_trip=False)
    assert breaker.open() is False
    assert breaker.current_state == STATE_OPEN


def test_manual_open_blocks_calls_until_timeout():
    breaker = CircuitBreaker(reset_timeout=60)
    breaker.open()
    assert breaker.state.name == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)


def test_manual_close_resets_failure_counter_and_state():
    breaker = CircuitBreaker(fail_max=5)
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    breaker.open()
    breaker.close()
    assert breaker.current_state == STATE_CLOSED
    assert breaker.state.name == STATE_CLOSED
    assert breaker.fail_counter == 0
    assert breaker.call(ok) == "ok"


def test_manual_half_open_sets_half_open_state():
    breaker = CircuitBreaker()
    breaker.half_open()
    assert breaker.current_state == STATE_HALF_OPEN
    assert isinstance(breaker.state, pybreaker.CircuitHalfOpenState)


def test_open_records_opened_at_timestamp():
    with freeze_time(START):
        storage = CircuitMemoryStorage(STATE_CLOSED)
        breaker = CircuitBreaker(fail_max=1, state_storage=storage)
        assert storage.opened_at is None
        with pytest.raises(CircuitBreakerError):
            breaker.call(fail)
        assert storage.opened_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_call_blocked_until_reset_timeout_elapses():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(breaker)
        frozen.tick(timedelta(seconds=9))
        with pytest.raises(CircuitBreakerError):
            breaker.call(ok)
        assert breaker.current_state == STATE_OPEN
        frozen.tick(timedelta(seconds=2))
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED


def test_call_allowed_exactly_when_timeout_elapses():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(breaker)
        frozen.tick(timedelta(seconds=10))
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED


def test_zero_reset_timeout_allows_immediate_trial_call():
    with freeze_time(START):
        breaker = CircuitBreaker(fail_max=1, reset_timeout=0)
        trip(breaker)
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED


def test_half_open_failure_reopens_with_circuit_breaker_error():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(breaker)
        frozen.tick(timedelta(seconds=11))
        with pytest.raises(CircuitBreakerError, match="Trial call failed"):
            breaker.call(fail)
        assert breaker.current_state == STATE_OPEN


def test_half_open_failure_without_new_error_reraises_original():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10, throw_new_error_on_trip=False)
        with pytest.raises(Boom):
            breaker.call(fail)
        frozen.tick(timedelta(seconds=11))
        with pytest.raises(Boom):
            breaker.call(fail)
        assert breaker.current_state == STATE_OPEN


def test_half_open_failure_restarts_the_timeout():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(breaker)
        frozen.tick(timedelta(seconds=10))
        with pytest.raises(CircuitBreakerError, match="Trial call failed"):
            breaker.call(fail)
        frozen.tick(timedelta(seconds=5))
        with pytest.raises(CircuitBreakerError, match="Timeout not elapsed"):
            breaker.call(ok)
        frozen.tick(timedelta(seconds=5))
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED


def test_success_threshold_requires_consecutive_successes_to_close():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=5, success_threshold=3)
        trip(breaker)
        frozen.tick(timedelta(seconds=6))
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_HALF_OPEN
        assert breaker.success_counter == 1
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_HALF_OPEN
        assert breaker.success_counter == 2
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED
        assert breaker.success_counter == 0


def test_failure_in_half_open_resets_success_counter():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=5, success_threshold=3)
        trip(breaker)
        frozen.tick(timedelta(seconds=6))
        breaker.call(ok)
        assert breaker.success_counter == 1
        with pytest.raises(CircuitBreakerError):
            breaker.call(fail)
        assert breaker.current_state == STATE_OPEN
        assert breaker.success_counter == 0


def test_closed_state_successes_do_not_touch_success_counter():
    breaker = CircuitBreaker(success_threshold=2)
    breaker.call(ok)
    breaker.call(ok)
    assert breaker.success_counter == 0
    assert breaker.current_state == STATE_CLOSED


def test_open_storage_without_opened_at_allows_trial_call():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_OPEN))
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_half_open_initial_storage_closes_after_successful_call():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_HALF_OPEN))
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


@pytest.mark.parametrize(
    ("age_seconds", "blocked"),
    [(10, True), (120, False)],
)
def test_preset_opened_at_controls_whether_call_is_blocked(age_seconds, blocked):
    storage = CircuitMemoryStorage(STATE_OPEN)
    storage.opened_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    breaker = CircuitBreaker(reset_timeout=60, state_storage=storage)
    if blocked:
        with pytest.raises(CircuitBreakerError):
            breaker.call(ok)
    else:
        assert breaker.call(ok) == "ok"
        assert breaker.current_state == STATE_CLOSED


def test_reset_timeout_setter_extends_open_period():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(breaker)
        frozen.tick(timedelta(seconds=11))
        breaker.reset_timeout = 100
        with pytest.raises(CircuitBreakerError, match="Timeout not elapsed"):
            breaker.call(ok)


def test_breakers_sharing_storage_see_each_others_open_state():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    first = CircuitBreaker(fail_max=1, state_storage=storage)
    second = CircuitBreaker(fail_max=1, state_storage=storage)
    with pytest.raises(CircuitBreakerError):
        first.call(fail)
    assert second.current_state == STATE_OPEN
    assert second.state.name == STATE_OPEN
    with pytest.raises(CircuitBreakerError, match="Timeout not elapsed"):
        second.call(ok)
