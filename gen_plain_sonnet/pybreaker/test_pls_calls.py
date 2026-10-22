"""Tests for calling through a breaker: decorators, context managers,
exception exclusion, generators and listener notifications."""

import sys
import types
from datetime import timedelta

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
    CircuitBreakerListener,
    CircuitMemoryStorage,
)

START = "2024-01-01 12:00:00"


class Boom(Exception):
    pass


class Fatal(BaseException):
    pass


class HttpError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def fail():
    raise Boom("boom")


def ok():
    return "ok"


class Recorder(CircuitBreakerListener):
    def __init__(self):
        self.events = []

    def before_call(self, cb, func, *args, **kwargs):
        self.events.append(("before_call", func, args, kwargs))

    def failure(self, cb, exc):
        self.events.append(("failure", exc))

    def success(self, cb):
        self.events.append(("success",))

    def state_change(self, cb, old_state, new_state):
        old = old_state.name if old_state is not None else None
        self.events.append(("state_change", old, new_state.name))

    @property
    def trace(self):
        return [e[0] if e[0] != "state_change" else f"{e[1]}->{e[2]}" for e in self.events]


def test_call_passes_positional_and_keyword_arguments():
    breaker = CircuitBreaker()

    def combine(a, b, sep="-"):
        return f"{a}{sep}{b}"

    assert breaker.call(combine, "x", "y", sep="+") == "x+y"


def test_call_returns_none_result():
    breaker = CircuitBreaker()
    assert breaker.call(lambda: None) is None


def test_base_exception_subclass_counts_as_failure():
    breaker = CircuitBreaker(fail_max=5)

    def raise_fatal():
        raise Fatal()

    with pytest.raises(Fatal):
        breaker.call(raise_fatal)
    assert breaker.fail_counter == 1


def test_decorator_without_parentheses_wraps_function_and_keeps_metadata():
    breaker = CircuitBreaker()

    @breaker
    def add(a, b=1):
        """Add numbers."""
        return a + b

    assert add(2, b=5) == 7
    assert add.__name__ == "add"
    assert add.__doc__ == "Add numbers."


def test_decorator_with_parentheses_wraps_function():
    breaker = CircuitBreaker()

    @breaker()
    def double(x):
        return x * 2

    assert double(21) == 42


def test_decorated_function_trips_breaker():
    breaker = CircuitBreaker(fail_max=2)

    @breaker
    def flaky():
        raise Boom("down")

    with pytest.raises(Boom):
        flaky()
    with pytest.raises(CircuitBreakerError):
        flaky()
    assert breaker.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError, match="Timeout not elapsed"):
        flaky()


def test_decorator_with_async_flag_false_uses_synchronous_call():
    breaker = CircuitBreaker()
    decorator = breaker(**{"__pybreaker_call_async": False})

    @decorator
    def triple(x):
        return x * 3

    assert triple(3) == 9


@pytest.mark.skipif(pybreaker.HAS_TORNADO_SUPPORT, reason="tornado is installed")
def test_decorator_with_async_flag_requires_tornado():
    breaker = CircuitBreaker()
    with pytest.raises(ImportError, match="tornado"):
        breaker(**{"__pybreaker_call_async": True})


def test_calling_context_manager_success_resets_failures():
    breaker = CircuitBreaker(fail_max=3)
    with pytest.raises(Boom):
        breaker.call(fail)
    ran = []
    with breaker.calling():
        ran.append(True)
    assert ran == [True]
    assert breaker.fail_counter == 0


def test_calling_context_manager_failure_counts_and_reraises():
    breaker = CircuitBreaker(fail_max=3)
    with pytest.raises(Boom):
        with breaker.calling():
            raise Boom("inside")
    assert breaker.fail_counter == 1
    assert breaker.current_state == STATE_CLOSED


def test_calling_context_manager_trips_breaker_at_threshold():
    breaker = CircuitBreaker(fail_max=2)
    with pytest.raises(Boom):
        with breaker.calling():
            raise Boom("one")
    with pytest.raises(CircuitBreakerError):
        with breaker.calling():
            raise Boom("two")
    assert breaker.current_state == STATE_OPEN


def test_calling_context_manager_rejects_when_open_without_running_body():
    breaker = CircuitBreaker(fail_max=1, reset_timeout=60)
    breaker.open()
    ran = []
    with pytest.raises(CircuitBreakerError):
        with breaker.calling():
            ran.append(True)
    assert ran == []


def test_generator_result_is_counted_only_when_consumed():
    def numbers():
        yield 1
        yield 2
        yield 3

    breaker = CircuitBreaker(fail_max=10)
    with pytest.raises(Boom):
        breaker.call(fail)
    result = breaker.call(numbers)
    assert isinstance(result, types.GeneratorType)
    assert breaker.fail_counter == 1
    assert list(result) == [1, 2, 3]
    assert breaker.fail_counter == 0


def test_generator_raising_midway_counts_failure_and_propagates():
    def broken():
        yield 1
        raise Boom("mid-stream")

    breaker = CircuitBreaker(fail_max=10)
    result = breaker.call(broken)
    with pytest.raises(Boom):
        list(result)
    assert breaker.fail_counter == 1


def test_nested_calls_through_same_breaker_do_not_deadlock():
    breaker = CircuitBreaker()
    assert breaker.call(lambda: breaker.call(lambda: 7)) == 7


def test_is_system_error_defaults_to_true():
    breaker = CircuitBreaker()
    assert breaker.is_system_error(Boom()) is True
    assert breaker.is_system_error(ValueError()) is True


def test_class_exclusion_covers_subclasses_only():
    breaker = CircuitBreaker(exclude=[LookupError])
    assert breaker.is_system_error(KeyError("k")) is False
    assert breaker.is_system_error(IndexError()) is False
    assert breaker.is_system_error(ValueError()) is True


def test_callable_exclusion_predicate():
    breaker = CircuitBreaker(exclude=[lambda e: isinstance(e, HttpError) and e.code < 500])
    assert breaker.is_system_error(HttpError(404)) is False
    assert breaker.is_system_error(HttpError(503)) is True
    assert breaker.is_system_error(ValueError()) is True


def test_excluded_exceptions_are_reraised_never_trip_and_reset_counter():
    breaker = CircuitBreaker(fail_max=2, exclude=[ValueError])
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.fail_counter == 1

    def bad_input():
        raise ValueError("business error")

    for _ in range(5):
        with pytest.raises(ValueError):
            breaker.call(bad_input)
    assert breaker.fail_counter == 0
    assert breaker.current_state == STATE_CLOSED


def test_excluded_exception_in_half_open_state_closes_circuit():
    with freeze_time(START) as frozen:
        breaker = CircuitBreaker(fail_max=1, reset_timeout=5, exclude=[ValueError])
        with pytest.raises(CircuitBreakerError):
            breaker.call(fail)
        frozen.tick(timedelta(seconds=6))

        def bad_input():
            raise ValueError("business error")

        with pytest.raises(ValueError):
            breaker.call(bad_input)
        assert breaker.current_state == STATE_CLOSED


def test_listener_before_call_receives_function_and_arguments():
    recorder = Recorder()
    breaker = CircuitBreaker(listeners=[recorder])

    def func(a, b=0):
        return a + b

    breaker.call(func, 1, b=2)
    assert recorder.events[0] == ("before_call", func, (1,), {"b": 2})


def test_listener_failure_receives_the_exception():
    recorder = Recorder()
    breaker = CircuitBreaker(fail_max=5, listeners=[recorder])
    with pytest.raises(Boom) as info:
        breaker.call(fail)
    failures = [e for e in recorder.events if e[0] == "failure"]
    assert len(failures) == 1
    assert failures[0][1] is info.value


def test_listener_trace_when_circuit_trips():
    recorder = Recorder()
    breaker = CircuitBreaker(fail_max=1, listeners=[recorder])
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert recorder.trace == ["before_call", "failure", "closed->open"]


def test_listener_trace_when_circuit_recovers():
    with freeze_time(START) as frozen:
        recorder = Recorder()
        breaker = CircuitBreaker(fail_max=1, reset_timeout=5, listeners=[recorder])
        with pytest.raises(CircuitBreakerError):
            breaker.call(fail)
        recorder.events.clear()
        frozen.tick(timedelta(seconds=6))
        assert breaker.call(ok) == "ok"
        assert recorder.trace == [
            "open->half-open",
            "before_call",
            "half-open->closed",
            "success",
        ]


def test_listener_not_notified_when_call_rejected_by_open_circuit():
    recorder = Recorder()
    breaker = CircuitBreaker(fail_max=1, reset_timeout=60, listeners=[recorder])
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    recorder.events.clear()
    with pytest.raises(CircuitBreakerError):
        breaker.call(ok)
    assert recorder.events == []


def test_listener_state_changes_for_manual_transitions():
    recorder = Recorder()
    breaker = CircuitBreaker()
    breaker.add_listener(recorder)
    breaker.open()
    breaker.half_open()
    breaker.close()
    assert recorder.trace == ["closed->open", "open->half-open", "half-open->closed"]


def test_excluded_exception_notifies_success_but_not_failure():
    recorder = Recorder()
    breaker = CircuitBreaker(exclude=[ValueError], listeners=[recorder])

    def bad_input():
        raise ValueError("business error")

    with pytest.raises(ValueError):
        breaker.call(bad_input)
    assert recorder.trace == ["before_call", "success"]


def test_listener_on_second_breaker_notified_of_external_state_change():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    first = CircuitBreaker(fail_max=1, state_storage=storage)
    recorder = Recorder()
    second = CircuitBreaker(fail_max=1, state_storage=storage, listeners=[recorder])
    with pytest.raises(CircuitBreakerError):
        first.call(fail)
    assert second.state.name == STATE_OPEN
    assert recorder.trace == ["closed->open"]
    assert STATE_HALF_OPEN not in recorder.trace
