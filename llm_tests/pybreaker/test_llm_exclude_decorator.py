"""Tests for exclude lists/predicates, decorator usage, and the calling() context manager."""
import functools
import time

import pytest

from pybreaker import (
    STATE_CLOSED,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
)


class ServiceError(Exception):
    pass


class BusinessError(Exception):
    pass


class SubBusinessError(BusinessError):
    pass


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


def raise_exc(exc):
    raise exc


# ---- exclusions -----------------------------------------------------------


def test_excluded_exception_does_not_count_as_failure():
    cb = CircuitBreaker(fail_max=2, exclude=[BusinessError])
    for _ in range(5):
        with pytest.raises(BusinessError):
            cb.call(raise_exc, BusinessError("nope"))
    assert cb.fail_counter == 0
    assert cb.current_state == STATE_CLOSED


def test_excluded_exception_subclass_is_excluded():
    cb = CircuitBreaker(fail_max=1, exclude=[BusinessError])
    with pytest.raises(SubBusinessError):
        cb.call(raise_exc, SubBusinessError())
    assert cb.current_state == STATE_CLOSED


def test_non_excluded_exception_still_counts():
    cb = CircuitBreaker(fail_max=3, exclude=[BusinessError])
    with pytest.raises(ServiceError):
        cb.call(raise_exc, ServiceError())
    assert cb.fail_counter == 1


def test_excluded_exception_resets_fail_counter():
    cb = CircuitBreaker(fail_max=5, exclude=[BusinessError])
    with pytest.raises(ServiceError):
        cb.call(raise_exc, ServiceError())
    assert cb.fail_counter == 1
    with pytest.raises(BusinessError):
        cb.call(raise_exc, BusinessError())
    assert cb.fail_counter == 0


def test_exclude_predicate():
    cb = CircuitBreaker(fail_max=2, exclude=[lambda e: isinstance(e, HttpError) and e.status < 500])
    for _ in range(4):
        with pytest.raises(HttpError):
            cb.call(raise_exc, HttpError(404))
    assert cb.fail_counter == 0
    assert cb.current_state == STATE_CLOSED


def test_exclude_predicate_false_counts_failure():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10, exclude=[lambda e: isinstance(e, HttpError) and e.status < 500])
    with pytest.raises(HttpError):
        cb.call(raise_exc, HttpError(503))
    with pytest.raises(CircuitBreakerError):
        cb.call(raise_exc, HttpError(500))
    assert cb.current_state == STATE_OPEN




def test_excluded_exceptions_property_returns_tuple():
    cb = CircuitBreaker(exclude=[BusinessError])
    assert cb.excluded_exceptions == (BusinessError,)
    assert isinstance(cb.excluded_exceptions, tuple)




def test_add_excluded_exception():
    cb = CircuitBreaker(fail_max=1)
    cb.add_excluded_exception(BusinessError)
    assert BusinessError in cb.excluded_exceptions
    with pytest.raises(BusinessError):
        cb.call(raise_exc, BusinessError())
    assert cb.current_state == STATE_CLOSED




def test_remove_excluded_exception():
    cb = CircuitBreaker(fail_max=5, exclude=[BusinessError])
    cb.remove_excluded_exception(BusinessError)
    assert cb.excluded_exceptions == ()
    with pytest.raises(BusinessError):
        cb.call(raise_exc, BusinessError())
    assert cb.fail_counter == 1




def test_is_system_error():
    cb = CircuitBreaker(exclude=[BusinessError])
    assert cb.is_system_error(ServiceError()) is True
    assert cb.is_system_error(BusinessError()) is False
    assert cb.is_system_error(SubBusinessError()) is False


def test_excluded_exception_in_half_open_counts_as_success():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, exclude=[BusinessError])
    with pytest.raises(CircuitBreakerError):
        cb.call(raise_exc, ServiceError())
    time.sleep(0.2)
    with pytest.raises(BusinessError):
        cb.call(raise_exc, BusinessError())
    assert cb.current_state == STATE_CLOSED


# ---- decorator ------------------------------------------------------------


def test_decorator_without_parentheses():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    @cb
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_decorator_with_parentheses():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    @cb()
    def add(a, b):
        return a + b

    assert add(a=4, b=5) == 9


def test_decorator_preserves_function_metadata():
    cb = CircuitBreaker()

    @cb
    def documented():
        """A docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring."


def test_decorator_trips_circuit():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    @cb
    def flaky():
        raise ServiceError()

    with pytest.raises(ServiceError):
        flaky()
    with pytest.raises(CircuitBreakerError):
        flaky()
    with pytest.raises(CircuitBreakerError):
        flaky()
    assert cb.current_state == STATE_OPEN


def test_decorator_recovers_after_timeout():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)
    state = {"healthy": False}

    @cb
    def service():
        if not state["healthy"]:
            raise ServiceError()
        return "up"

    with pytest.raises(CircuitBreakerError):
        service()
    state["healthy"] = True
    with pytest.raises(CircuitBreakerError):
        service()
    time.sleep(0.2)
    assert service() == "up"
    assert cb.current_state == STATE_CLOSED


def test_two_functions_share_the_same_breaker():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    @cb
    def a():
        raise ServiceError()

    @cb
    def b():
        return "b"

    with pytest.raises(ServiceError):
        a()
    with pytest.raises(CircuitBreakerError):
        a()
    with pytest.raises(CircuitBreakerError):
        b()






# ---- calling() context manager -------------------------------------------


def test_calling_context_manager_success():
    cb = CircuitBreaker(fail_max=2)
    with cb.calling():
        pass
    assert cb.fail_counter == 0
    assert cb.current_state == STATE_CLOSED


def test_calling_context_manager_counts_failure_and_reraises():
    cb = CircuitBreaker(fail_max=3)
    with pytest.raises(ServiceError):
        with cb.calling():
            raise ServiceError()
    assert cb.fail_counter == 1


def test_calling_context_manager_trips_circuit():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)
    for expected in (ServiceError, CircuitBreakerError):
        with pytest.raises(expected):
            with cb.calling():
                raise ServiceError()
    assert cb.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        with cb.calling():
            pass
