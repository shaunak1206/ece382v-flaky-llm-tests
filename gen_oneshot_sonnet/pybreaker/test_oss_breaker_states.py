"""Tests for CircuitBreaker configuration and the closed/open/half-open state machine."""

from unittest import mock

import pytest
from freezegun import freeze_time

import pybreaker
from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
)


class Boom(Exception):
    pass


def fail():
    raise Boom("boom")


def ok():
    return "ok"


def trip(breaker):
    """Drive a closed breaker to the open state with failing calls."""
    for _ in range(breaker.fail_max):
        with pytest.raises((Boom, CircuitBreakerError)):
            breaker.call(fail)


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


# ------------------------------------------------------------ configuration


def test_state_constants():
    assert STATE_CLOSED == "closed"
    assert STATE_OPEN == "open"
    assert STATE_HALF_OPEN == "half-open"


def test_default_configuration():
    breaker = CircuitBreaker()
    assert breaker.fail_max == 5
    assert breaker.reset_timeout == 60
    assert breaker.success_threshold == 1
    assert breaker.name is None
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0
    assert breaker.success_counter == 0
    assert breaker.listeners == ()
    assert breaker.excluded_exceptions == ()


def test_constructor_parameters_are_stored():
    breaker = CircuitBreaker(fail_max=3, reset_timeout=10, success_threshold=2, name="db")
    assert breaker.fail_max == 3
    assert breaker.reset_timeout == 10
    assert breaker.success_threshold == 2
    assert breaker.name == "db"


def test_property_setters():
    breaker = CircuitBreaker()
    breaker.fail_max = 9
    breaker.reset_timeout = 1.5
    breaker.success_threshold = 4
    breaker.name = "renamed"
    assert (breaker.fail_max, breaker.reset_timeout) == (9, 1.5)
    assert (breaker.success_threshold, breaker.name) == (4, "renamed")


def test_state_object_exposes_name():
    breaker = CircuitBreaker()
    assert breaker.state.name == STATE_CLOSED


def test_unknown_state_in_storage_raises_value_error():
    with pytest.raises(ValueError, match="Unknown state"):
        CircuitBreaker(state_storage=CircuitMemoryStorage("bogus"))


# ------------------------------------------------------------- closed state


def test_successful_calls_keep_circuit_closed():
    breaker = CircuitBreaker(fail_max=2)
    for _ in range(10):
        assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0


def test_failure_below_threshold_reraises_original_and_counts():
    breaker = CircuitBreaker(fail_max=3)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.fail_counter == 1
    assert breaker.current_state == STATE_CLOSED


def test_circuit_opens_when_fail_max_reached():
    breaker = CircuitBreaker(fail_max=3)
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(CircuitBreakerError, match="Failures threshold reached"):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN
    assert breaker.fail_counter == 3


def test_fail_max_of_one_trips_on_first_failure():
    breaker = CircuitBreaker(fail_max=1)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_success_resets_consecutive_failure_count():
    breaker = CircuitBreaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.fail_counter == 2
    breaker.call(ok)
    assert breaker.fail_counter == 0
    for _ in range(2):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.current_state == STATE_CLOSED


def test_fail_max_setter_takes_effect_immediately():
    breaker = CircuitBreaker(fail_max=10)
    breaker.fail_max = 2
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


# --------------------------------------------------------------- open state


def test_open_circuit_rejects_calls_without_running_function():
    breaker = CircuitBreaker(fail_max=1)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    func = mock.Mock(return_value="never")
    with pytest.raises(CircuitBreakerError, match="Timeout not elapsed yet"):
        breaker.call(func)
    func.assert_not_called()


def test_circuit_stays_open_until_reset_timeout_elapses(frozen):
    breaker = CircuitBreaker(fail_max=2, reset_timeout=60)
    trip(breaker)
    assert breaker.current_state == STATE_OPEN

    frozen.move_to("2024-01-01 12:00:30")
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)

    frozen.move_to("2024-01-01 12:00:59")
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)
    assert breaker.current_state == STATE_OPEN


def test_call_after_reset_timeout_closes_circuit_on_success(frozen):
    breaker = CircuitBreaker(fail_max=2, reset_timeout=60)
    trip(breaker)

    frozen.move_to("2024-01-01 12:01:01")
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0


def test_state_still_reports_open_after_timeout_until_next_call(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=5)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    frozen.move_to("2024-01-01 12:10:00")
    assert breaker.current_state == STATE_OPEN
    breaker.call(ok)
    assert breaker.current_state == STATE_CLOSED


def test_trial_call_runs_while_circuit_is_half_open(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=5)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    seen = []

    def probe():
        seen.append(breaker.current_state)
        return "probed"

    frozen.move_to("2024-01-01 12:00:06")
    assert breaker.call(probe) == "probed"
    assert seen == [STATE_HALF_OPEN]


def test_failed_trial_call_reopens_circuit_and_restarts_timeout(frozen):
    breaker = CircuitBreaker(fail_max=2, reset_timeout=30)
    trip(breaker)
    trial = mock.Mock(side_effect=Boom)

    frozen.move_to("2024-01-01 12:00:31")
    with pytest.raises(CircuitBreakerError, match="Trial call failed"):
        breaker.call(trial)
    assert breaker.current_state == STATE_OPEN
    assert trial.call_count == 1

    # The timeout restarted at 12:00:31, so this call is rejected untouched.
    with pytest.raises(CircuitBreakerError, match="Timeout not elapsed yet"):
        breaker.call(trial)
    assert trial.call_count == 1

    frozen.move_to("2024-01-01 12:01:02")
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_changing_reset_timeout_applies_to_already_open_circuit(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=600)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    frozen.move_to("2024-01-01 12:00:20")
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)
    breaker.reset_timeout = 10
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_breaker_built_on_open_storage_without_opened_at_half_opens_on_call():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_OPEN))
    assert breaker.current_state == STATE_OPEN
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


# ---------------------------------------------------------- success_threshold


def test_success_threshold_requires_multiple_trial_successes(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=10, success_threshold=3)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)

    frozen.move_to("2024-01-01 12:00:11")
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_HALF_OPEN
    assert breaker.success_counter == 1

    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_HALF_OPEN
    assert breaker.success_counter == 2

    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED
    assert breaker.success_counter == 0


def test_failure_while_half_open_resets_success_counter(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=10, success_threshold=2)
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)

    frozen.move_to("2024-01-01 12:00:11")
    breaker.call(ok)
    assert breaker.success_counter == 1
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN
    assert breaker.success_counter == 0


# ------------------------------------------------- manual open/half_open/close


def test_manual_open_rejects_calls_and_returns_true():
    breaker = CircuitBreaker()
    assert breaker.open() is True
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)


def test_manual_half_open_then_success_closes():
    breaker = CircuitBreaker()
    breaker.half_open()
    assert breaker.current_state == STATE_HALF_OPEN
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_manual_half_open_then_failure_opens():
    breaker = CircuitBreaker()
    breaker.half_open()
    with pytest.raises(CircuitBreakerError, match="Trial call failed"):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_manual_close_resets_failures_and_allows_calls():
    breaker = CircuitBreaker(fail_max=2)
    trip(breaker)
    assert breaker.current_state == STATE_OPEN
    breaker.close()
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0
    assert breaker.call(ok) == "ok"


def test_manual_open_then_timeout_allows_recovery(frozen):
    breaker = CircuitBreaker(reset_timeout=20)
    breaker.open()
    frozen.move_to("2024-01-01 12:00:21")
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_public_exports():
    for name in pybreaker.__all__:
        assert hasattr(pybreaker, name)
