import threading

import pybreaker
import pytest


class Boom(Exception):
    pass


def boom():
    raise Boom("boom")


def ok():
    return "ok"


def test_call_returns_function_result():
    cb = pybreaker.CircuitBreaker()
    assert cb.call(ok) == "ok"


def test_call_forwards_positional_and_keyword_arguments():
    cb = pybreaker.CircuitBreaker()

    def add(a, b, c=0):
        return a + b + c

    assert cb.call(add, 1, 2, c=3) == 6


def test_call_propagates_original_exception_below_threshold():
    cb = pybreaker.CircuitBreaker(fail_max=3)
    with pytest.raises(Boom, match="boom"):
        cb.call(boom)


def test_call_returning_none():
    cb = pybreaker.CircuitBreaker()
    assert cb.call(lambda: None) is None


def test_decorator_returns_result_and_forwards_arguments():
    cb = pybreaker.CircuitBreaker()

    @cb
    def multiply(a, b=1):
        return a * b

    assert multiply(3, b=4) == 12


def test_decorator_preserves_function_name():
    cb = pybreaker.CircuitBreaker()

    @cb
    def my_named_function():
        return 1

    assert my_named_function.__name__ == "my_named_function"


def test_decorator_counts_failures_and_trips():
    cb = pybreaker.CircuitBreaker(fail_max=2)

    @cb
    def failing():
        raise Boom("x")

    with pytest.raises(Boom):
        failing()
    assert cb.fail_counter == 1
    with pytest.raises(pybreaker.CircuitBreakerError):
        failing()
    assert cb.current_state == "open"


def test_decorated_functions_share_breaker_state():
    cb = pybreaker.CircuitBreaker(fail_max=1)

    @cb
    def failing():
        raise Boom("x")

    @cb
    def healthy():
        return "fine"

    with pytest.raises(pybreaker.CircuitBreakerError):
        failing()
    with pytest.raises(pybreaker.CircuitBreakerError):
        healthy()


def test_context_manager_success_does_not_change_state():
    cb = pybreaker.CircuitBreaker(fail_max=2)
    with cb.calling():
        pass
    assert cb.current_state == "closed"
    assert cb.fail_counter == 0


def test_context_manager_propagates_exception_and_counts_failure():
    cb = pybreaker.CircuitBreaker(fail_max=3)
    with pytest.raises(Boom):
        with cb.calling():
            raise Boom("inside")
    assert cb.fail_counter == 1


def test_context_manager_trips_circuit():
    cb = pybreaker.CircuitBreaker(fail_max=2)
    with pytest.raises(Boom):
        with cb.calling():
            raise Boom("one")
    with pytest.raises(pybreaker.CircuitBreakerError):
        with cb.calling():
            raise Boom("two")
    assert cb.current_state == "open"


def test_context_manager_blocked_when_circuit_open():
    cb = pybreaker.CircuitBreaker()
    cb.open()
    executed = []
    with pytest.raises(pybreaker.CircuitBreakerError):
        with cb.calling():
            executed.append(True)
    assert executed == []


def test_concurrent_successful_calls_leave_circuit_closed():
    cb = pybreaker.CircuitBreaker(fail_max=2)
    results = []

    def worker():
        for _ in range(20):
            results.append(cb.call(ok))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["ok"] * 100
    assert cb.current_state == "closed"
    assert cb.fail_counter == 0
