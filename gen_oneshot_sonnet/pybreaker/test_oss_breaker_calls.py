"""Tests for call(), the decorator form, calling(), call_async and misc call behavior."""

import threading
from unittest import mock

import pytest
from freezegun import freeze_time

import pybreaker
from pybreaker import (
    STATE_CLOSED,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
)


class Boom(Exception):
    pass


def fail():
    raise Boom("boom")


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


def test_call_passes_args_and_kwargs_and_returns_result():
    breaker = CircuitBreaker()
    func = mock.Mock(return_value=[1, 2, 3])
    assert breaker.call(func, "a", 2, key="v") == [1, 2, 3]
    func.assert_called_once_with("a", 2, key="v")


def test_call_returns_none_and_falsy_results_unchanged():
    breaker = CircuitBreaker()
    assert breaker.call(lambda: None) is None
    assert breaker.call(lambda: 0) == 0
    assert breaker.call(lambda: "") == ""


def test_circuit_breaker_error_is_a_plain_exception():
    assert issubclass(CircuitBreakerError, Exception)
    assert not issubclass(CircuitBreakerError, Boom)
    assert str(CircuitBreakerError("custom")) == "custom"


def test_decorator_wraps_function_and_preserves_metadata():
    breaker = CircuitBreaker()

    @breaker
    def add(a, b=0):
        """Add two numbers."""
        return a + b

    assert add(1, b=2) == 3
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."


def test_decorator_with_parentheses():
    breaker = CircuitBreaker()

    @breaker()
    def double(x):
        return x * 2

    assert double(21) == 42


def test_decorated_function_trips_the_breaker_and_is_then_skipped():
    breaker = CircuitBreaker(fail_max=2)
    calls = []

    @breaker
    def flaky():
        calls.append(1)
        raise Boom("down")

    with pytest.raises(Boom):
        flaky()
    with pytest.raises(CircuitBreakerError):
        flaky()
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        flaky()
    assert len(calls) == 2


def test_decorated_function_recovers_after_reset_timeout(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=30)
    healthy = {"value": False}

    @breaker
    def service():
        if not healthy["value"]:
            raise Boom("down")
        return "up"

    with pytest.raises(CircuitBreakerError):
        service()
    healthy["value"] = True
    with pytest.raises(CircuitBreakerError):
        service()

    frozen.move_to("2024-01-01 12:00:31")
    assert service() == "up"
    assert breaker.current_state == STATE_CLOSED


def test_breaker_shared_by_two_functions_shares_failure_count():
    breaker = CircuitBreaker(fail_max=2)

    @breaker
    def first():
        raise Boom("first")

    @breaker
    def second():
        raise Boom("second")

    with pytest.raises(Boom):
        first()
    with pytest.raises(CircuitBreakerError):
        second()
    assert breaker.current_state == STATE_OPEN


def test_throw_new_error_on_trip_false_raises_original_exception():
    breaker = CircuitBreaker(fail_max=2, throw_new_error_on_trip=False)
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)


def test_throw_new_error_on_trip_false_open_returns_false(frozen):
    breaker = CircuitBreaker(fail_max=1, reset_timeout=5, throw_new_error_on_trip=False)
    assert breaker.open() is False
    frozen.move_to("2024-01-01 12:00:06")
    breaker.half_open()
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.current_state == STATE_OPEN


def test_calling_context_manager_success_keeps_circuit_closed():
    breaker = CircuitBreaker(fail_max=2)
    with breaker.calling():
        value = 1 + 1
    assert value == 2
    assert breaker.fail_counter == 0
    assert breaker.current_state == STATE_CLOSED


def test_calling_context_manager_counts_failures_and_trips():
    breaker = CircuitBreaker(fail_max=2)
    with pytest.raises(Boom):
        with breaker.calling():
            raise Boom("first")
    assert breaker.fail_counter == 1
    with pytest.raises(CircuitBreakerError):
        with breaker.calling():
            raise Boom("second")
    assert breaker.current_state == STATE_OPEN


def test_calling_context_manager_rejects_when_open():
    breaker = CircuitBreaker()
    breaker.open()
    with pytest.raises(CircuitBreakerError):
        with breaker.calling():
            pytest.fail("body must not run while the circuit is open")


def test_concurrent_failures_are_counted_exactly():
    breaker = CircuitBreaker(fail_max=10_000)

    def worker():
        for _ in range(20):
            try:
                breaker.call(fail)
            except Boom:
                pass

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert breaker.fail_counter == 200
    assert breaker.current_state == STATE_CLOSED


@pytest.mark.skipif(pybreaker.HAS_TORNADO_SUPPORT, reason="tornado is installed")
def test_async_decorator_without_tornado_raises_import_error():
    breaker = CircuitBreaker()
    with pytest.raises(ImportError):
        breaker(**{"__pybreaker_call_async": True})


def test_call_async_returns_result_and_trips():
    gen = pytest.importorskip("tornado.gen")
    ioloop = pytest.importorskip("tornado.ioloop")
    breaker = CircuitBreaker(fail_max=2)

    @gen.coroutine
    def double(x):
        raise gen.Return(x * 2)

    @gen.coroutine
    def broken():
        raise Boom("async boom")
        yield  # pragma: no cover

    loop = ioloop.IOLoop.current()
    assert loop.run_sync(lambda: breaker.call_async(double, 4)) == 8
    with pytest.raises(Boom):
        loop.run_sync(lambda: breaker.call_async(broken))
    with pytest.raises(CircuitBreakerError):
        loop.run_sync(lambda: breaker.call_async(broken))
    assert breaker.current_state == STATE_OPEN
