"""Tests for CircuitBreaker construction, state transitions, and reset_timeout."""
import time

import pytest

import pybreaker
from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
)


class ServiceError(Exception):
    pass


def failing():
    raise ServiceError("boom")


def succeeding():
    return "ok"


@pytest.fixture
def breaker():
    return CircuitBreaker(fail_max=3, reset_timeout=0.2)


def trip(cb):
    """Fail until the breaker opens."""
    for _ in range(cb.fail_max):
        with pytest.raises((ServiceError, CircuitBreakerError)):
            cb.call(failing)


def test_default_parameters():
    cb = CircuitBreaker()
    assert cb.fail_max == 5
    assert cb.reset_timeout == 60
    assert cb.success_threshold == 1
    assert cb.name is None
    assert cb.current_state == STATE_CLOSED






def test_successful_call_returns_value(breaker):
    assert breaker.call(lambda a, b=0: a + b, 2, b=3) == 5
    assert breaker.fail_counter == 0


def test_failed_call_reraises_original_exception(breaker):
    with pytest.raises(ServiceError):
        breaker.call(failing)
    assert breaker.fail_counter == 1
    assert breaker.current_state == STATE_CLOSED


def test_circuit_opens_after_fail_max_failures(breaker):
    for _ in range(breaker.fail_max - 1):
        with pytest.raises(ServiceError):
            breaker.call(failing)
    assert breaker.current_state == STATE_CLOSED
    with pytest.raises(CircuitBreakerError):
        breaker.call(failing)
    assert breaker.current_state == STATE_OPEN
    assert breaker.fail_counter == breaker.fail_max


def test_success_resets_fail_counter(breaker):
    with pytest.raises(ServiceError):
        breaker.call(failing)
    with pytest.raises(ServiceError):
        breaker.call(failing)
    assert breaker.fail_counter == 2
    breaker.call(succeeding)
    assert breaker.fail_counter == 0


def test_open_circuit_rejects_calls_without_running_function(breaker):
    trip(breaker)
    calls = []
    with pytest.raises(CircuitBreakerError):
        breaker.call(lambda: calls.append(1))
    assert calls == []
    assert breaker.current_state == STATE_OPEN




def test_throw_new_error_on_trip_false_raises_original():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10, throw_new_error_on_trip=False)
    with pytest.raises(ServiceError):
        cb.call(failing)
    with pytest.raises(ServiceError):
        cb.call(failing)
    assert cb.current_state == STATE_OPEN
    # Subsequent calls while open still raise CircuitBreakerError.
    with pytest.raises(CircuitBreakerError):
        cb.call(succeeding)


def test_circuit_stays_open_before_timeout():
    cb = CircuitBreaker(fail_max=1, reset_timeout=5)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.05)
    with pytest.raises(CircuitBreakerError):
        cb.call(succeeding)
    assert cb.current_state == STATE_OPEN


def test_circuit_closes_after_timeout_and_successful_call(breaker):
    trip(breaker)
    assert breaker.current_state == STATE_OPEN
    time.sleep(0.3)
    assert breaker.call(succeeding) == "ok"
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0


def test_circuit_reopens_if_trial_call_fails(breaker):
    trip(breaker)
    time.sleep(0.3)
    with pytest.raises(CircuitBreakerError):
        breaker.call(failing)
    assert breaker.current_state == STATE_OPEN


def test_trial_call_failure_with_no_new_error_raises_original():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, throw_new_error_on_trip=False)
    with pytest.raises(ServiceError):
        cb.call(failing)
    time.sleep(0.2)
    with pytest.raises(ServiceError):
        cb.call(failing)
    assert cb.current_state == STATE_OPEN


def test_state_becomes_half_open_when_trial_call_is_running():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)
    seen = []

    def probe():
        seen.append(cb.current_state)
        return "done"

    cb.call(probe)
    assert seen == [STATE_HALF_OPEN]
    assert cb.current_state == STATE_CLOSED


def test_reset_timeout_can_be_changed_at_runtime(breaker):
    breaker.reset_timeout = 30
    assert breaker.reset_timeout == 30
    trip(breaker)
    time.sleep(0.3)
    with pytest.raises(CircuitBreakerError):
        breaker.call(succeeding)


def test_fail_max_setter_changes_threshold():
    cb = CircuitBreaker(fail_max=10, reset_timeout=10)
    cb.fail_max = 2
    assert cb.fail_max == 2
    with pytest.raises(ServiceError):
        cb.call(failing)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert cb.current_state == STATE_OPEN




def test_manual_open_blocks_calls(breaker):
    breaker.open()
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        breaker.call(succeeding)




def test_manual_close_after_trip(breaker):
    trip(breaker)
    breaker.close()
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0
    assert breaker.call(succeeding) == "ok"


def test_manual_half_open_then_success_closes(breaker):
    breaker.half_open()
    assert breaker.current_state == STATE_HALF_OPEN
    assert breaker.state.name == STATE_HALF_OPEN
    assert breaker.call(succeeding) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_manual_half_open_then_failure_opens(breaker):
    breaker.half_open()
    with pytest.raises(CircuitBreakerError):
        breaker.call(failing)
    assert breaker.current_state == STATE_OPEN




def test_unknown_state_raises_value_error(breaker):
    with pytest.raises(ValueError):
        breaker.state = "bogus"






