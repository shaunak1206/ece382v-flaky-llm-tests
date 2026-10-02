"""Tests for excluded exceptions and exclusion predicates."""

import pytest
from freezegun import freeze_time

from pybreaker import (
    STATE_CLOSED,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
)


class BusinessError(Exception):
    pass


class SpecificBusinessError(BusinessError):
    pass


class SystemFailure(Exception):
    pass


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def raiser(exc):
    def _raise():
        raise exc

    return _raise


def client_error(exc):
    return isinstance(exc, HttpError) and exc.status < 500


def test_excluded_exception_does_not_trip_the_circuit():
    breaker = CircuitBreaker(fail_max=2, exclude=[BusinessError])
    for _ in range(10):
        with pytest.raises(BusinessError):
            breaker.call(raiser(BusinessError("nope")))
    assert breaker.current_state == STATE_CLOSED
    assert breaker.fail_counter == 0


def test_excluded_exception_is_reraised_unchanged():
    breaker = CircuitBreaker(exclude=[BusinessError])
    original = BusinessError("keep me")
    with pytest.raises(BusinessError) as excinfo:
        breaker.call(raiser(original))
    assert excinfo.value is original


def test_excluded_exception_resets_failure_counter():
    breaker = CircuitBreaker(fail_max=3, exclude=[BusinessError])
    for _ in range(2):
        with pytest.raises(SystemFailure):
            breaker.call(raiser(SystemFailure()))
    assert breaker.fail_counter == 2
    with pytest.raises(BusinessError):
        breaker.call(raiser(BusinessError()))
    assert breaker.fail_counter == 0


def test_subclasses_of_excluded_exception_are_excluded():
    breaker = CircuitBreaker(fail_max=1, exclude=[BusinessError])
    with pytest.raises(SpecificBusinessError):
        breaker.call(raiser(SpecificBusinessError()))
    assert breaker.current_state == STATE_CLOSED


def test_unlisted_exceptions_still_count_as_failures():
    breaker = CircuitBreaker(fail_max=2, exclude=[BusinessError])
    with pytest.raises(SystemFailure):
        breaker.call(raiser(SystemFailure()))
    with pytest.raises(CircuitBreakerError):
        breaker.call(raiser(SystemFailure()))
    assert breaker.current_state == STATE_OPEN


def test_exclusion_predicate_decides_per_exception_instance():
    breaker = CircuitBreaker(fail_max=2, exclude=[client_error])
    for _ in range(5):
        with pytest.raises(HttpError):
            breaker.call(raiser(HttpError(404)))
    assert breaker.fail_counter == 0
    assert breaker.current_state == STATE_CLOSED

    with pytest.raises(HttpError):
        breaker.call(raiser(HttpError(503)))
    with pytest.raises(CircuitBreakerError):
        breaker.call(raiser(HttpError(503)))
    assert breaker.current_state == STATE_OPEN


def test_classes_and_predicates_can_be_mixed():
    breaker = CircuitBreaker(fail_max=1, exclude=(BusinessError, client_error))
    assert breaker.is_system_error(BusinessError()) is False
    assert breaker.is_system_error(HttpError(400)) is False
    assert breaker.is_system_error(HttpError(500)) is True
    assert breaker.is_system_error(SystemFailure()) is True


def test_add_and_remove_excluded_exceptions():
    breaker = CircuitBreaker()
    assert breaker.excluded_exceptions == ()
    breaker.add_excluded_exception(BusinessError)
    breaker.add_excluded_exceptions(KeyError, ValueError)
    assert breaker.excluded_exceptions == (BusinessError, KeyError, ValueError)
    assert breaker.is_system_error(KeyError("k")) is False

    breaker.remove_excluded_exception(KeyError)
    assert breaker.excluded_exceptions == (BusinessError, ValueError)
    assert breaker.is_system_error(KeyError("k")) is True


def test_remove_unknown_excluded_exception_raises_value_error():
    breaker = CircuitBreaker()
    with pytest.raises(ValueError):
        breaker.remove_excluded_exception(BusinessError)


def test_excluded_exception_during_trial_call_closes_circuit():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=60, exclude=[BusinessError])
        with pytest.raises(CircuitBreakerError):
            breaker.call(raiser(SystemFailure()))
        assert breaker.current_state == STATE_OPEN

        frozen.move_to("2024-01-01 12:01:01")
        with pytest.raises(BusinessError):
            breaker.call(raiser(BusinessError()))
        assert breaker.current_state == STATE_CLOSED
