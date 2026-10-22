import pybreaker
import pytest


class Boom(Exception):
    pass


class BusinessError(Exception):
    pass


class SpecificBusinessError(BusinessError):
    pass


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status_code = status


def test_no_excluded_exceptions_by_default():
    assert tuple(pybreaker.CircuitBreaker().excluded_exceptions) == ()


def test_excluded_exceptions_is_a_tuple():
    cb = pybreaker.CircuitBreaker(exclude=[BusinessError])
    assert isinstance(cb.excluded_exceptions, tuple)
    assert BusinessError in cb.excluded_exceptions


def test_is_system_error_true_by_default():
    cb = pybreaker.CircuitBreaker()
    assert cb.is_system_error(Boom("x")) is True


def test_is_system_error_false_for_excluded_type():
    cb = pybreaker.CircuitBreaker(exclude=[BusinessError])
    assert cb.is_system_error(BusinessError("x")) is False
    assert cb.is_system_error(Boom("x")) is True


def test_subclasses_of_excluded_exception_are_excluded():
    cb = pybreaker.CircuitBreaker(exclude=[BusinessError])
    assert cb.is_system_error(SpecificBusinessError("x")) is False


def test_excluded_exception_does_not_open_circuit_and_propagates():
    cb = pybreaker.CircuitBreaker(fail_max=2, exclude=[BusinessError])

    def business():
        raise BusinessError("invalid")

    for _ in range(10):
        with pytest.raises(BusinessError):
            cb.call(business)
    assert cb.current_state == "closed"
    assert cb.fail_counter == 0


def test_add_excluded_exception():
    cb = pybreaker.CircuitBreaker()
    cb.add_excluded_exception(BusinessError)
    assert cb.is_system_error(BusinessError("x")) is False
    assert BusinessError in cb.excluded_exceptions


def test_add_excluded_exceptions_multiple():
    cb = pybreaker.CircuitBreaker()
    cb.add_excluded_exceptions(BusinessError, KeyError)
    assert cb.is_system_error(BusinessError("x")) is False
    assert cb.is_system_error(KeyError("k")) is False
    assert cb.is_system_error(Boom("x")) is True


def test_remove_excluded_exception():
    cb = pybreaker.CircuitBreaker(exclude=[BusinessError])
    cb.remove_excluded_exception(BusinessError)
    assert cb.is_system_error(BusinessError("x")) is True
    assert BusinessError not in cb.excluded_exceptions


def test_callable_filter_decides_per_exception():
    cb = pybreaker.CircuitBreaker(
        exclude=[lambda e: isinstance(e, HttpError) and e.status_code < 500]
    )
    assert cb.is_system_error(HttpError(404)) is False
    assert cb.is_system_error(HttpError(503)) is True


def test_mixed_types_and_callables():
    cb = pybreaker.CircuitBreaker(
        exclude=[BusinessError, lambda e: isinstance(e, HttpError) and e.status_code < 500]
    )
    assert cb.is_system_error(BusinessError("x")) is False
    assert cb.is_system_error(HttpError(400)) is False
    assert cb.is_system_error(HttpError(500)) is True
    assert cb.is_system_error(Boom("x")) is True


def test_non_excluded_failures_still_trip_when_others_excluded():
    cb = pybreaker.CircuitBreaker(fail_max=2, exclude=[BusinessError])
    with pytest.raises(Boom):
        cb.call(_raise, Boom("a"))
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(_raise, Boom("b"))
    assert cb.current_state == "open"


def _raise(exc):
    raise exc
