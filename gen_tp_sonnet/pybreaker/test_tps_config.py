import pybreaker
import pytest


class Boom(Exception):
    pass


def boom():
    raise Boom("boom")


def ok():
    return "ok"


def test_default_fail_max_is_five():
    assert pybreaker.CircuitBreaker().fail_max == 5


def test_default_reset_timeout_is_sixty():
    assert pybreaker.CircuitBreaker().reset_timeout == 60


def test_default_success_threshold_is_one():
    assert pybreaker.CircuitBreaker().success_threshold == 1


def test_default_state_is_closed():
    assert pybreaker.CircuitBreaker().current_state == "closed"


def test_default_counters_are_zero():
    cb = pybreaker.CircuitBreaker()
    assert cb.fail_counter == 0
    assert cb.success_counter == 0


def test_default_name_is_none():
    assert pybreaker.CircuitBreaker().name is None


def test_constructor_values_are_stored():
    cb = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=12.5, success_threshold=4, name="db")
    assert cb.fail_max == 3
    assert cb.reset_timeout == 12.5
    assert cb.success_threshold == 4
    assert cb.name == "db"


def test_fail_max_setter():
    cb = pybreaker.CircuitBreaker()
    cb.fail_max = 9
    assert cb.fail_max == 9


def test_reset_timeout_setter():
    cb = pybreaker.CircuitBreaker()
    cb.reset_timeout = 7
    assert cb.reset_timeout == 7


def test_success_threshold_setter():
    cb = pybreaker.CircuitBreaker()
    cb.success_threshold = 3
    assert cb.success_threshold == 3


def test_name_setter():
    cb = pybreaker.CircuitBreaker(name="first")
    cb.name = "second"
    assert cb.name == "second"


def test_circuit_breaker_error_is_an_exception():
    assert issubclass(pybreaker.CircuitBreakerError, Exception)
    with pytest.raises(Exception):
        raise pybreaker.CircuitBreakerError("open")


def test_lowered_fail_max_takes_effect():
    cb = pybreaker.CircuitBreaker(fail_max=5)
    cb.fail_max = 2
    with pytest.raises(Boom):
        cb.call(boom)
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(boom)
    assert cb.current_state == "open"
