"""Tests for construction, configuration, storage and bookkeeping of pybreaker."""

import sys

import pytest

try:
    import pybreaker
except ImportError:  # pragma: no cover - fallback for src layout
    sys.path.insert(0, "/Users/shaunkapoor/ece382v-flaky-llm-tests/subjects/pybreaker/src")
    import pybreaker

from datetime import datetime, timezone

from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerListener,
    CircuitMemoryStorage,
)


class Boom(Exception):
    pass


def ok():
    return "ok"


def test_state_constants_values():
    assert STATE_OPEN == "open"
    assert STATE_CLOSED == "closed"
    assert STATE_HALF_OPEN == "half-open"


def test_all_exports_public_names():
    exported = set(pybreaker.__all__)
    for name in (
        "CircuitBreaker",
        "CircuitBreakerListener",
        "CircuitBreakerError",
        "CircuitMemoryStorage",
        "CircuitRedisStorage",
        "STATE_OPEN",
        "STATE_CLOSED",
        "STATE_HALF_OPEN",
    ):
        assert name in exported


def test_default_parameters():
    breaker = CircuitBreaker()
    assert breaker.fail_max == 5
    assert breaker.reset_timeout == 60
    assert breaker.success_threshold == 1
    assert breaker.name is None
    assert breaker.fail_counter == 0
    assert breaker.success_counter == 0
    assert breaker.current_state == STATE_CLOSED
    assert breaker.state.name == STATE_CLOSED


def test_custom_constructor_parameters():
    breaker = CircuitBreaker(fail_max=2, reset_timeout=7.5, success_threshold=3, name="svc")
    assert breaker.fail_max == 2
    assert breaker.reset_timeout == 7.5
    assert breaker.success_threshold == 3
    assert breaker.name == "svc"


def test_property_setters_update_values():
    breaker = CircuitBreaker()
    breaker.fail_max = 10
    breaker.reset_timeout = 2.5
    breaker.success_threshold = 4
    breaker.name = "renamed"
    assert breaker.fail_max == 10
    assert breaker.reset_timeout == 2.5
    assert breaker.success_threshold == 4
    assert breaker.name == "renamed"


def test_initial_state_closed_uses_closed_state_object():
    breaker = CircuitBreaker()
    assert isinstance(breaker.state, pybreaker.CircuitClosedState)


def test_initial_state_taken_from_open_storage():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_OPEN))
    assert breaker.current_state == STATE_OPEN
    assert isinstance(breaker.state, pybreaker.CircuitOpenState)


def test_initial_state_taken_from_half_open_storage():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_HALF_OPEN))
    assert breaker.current_state == STATE_HALF_OPEN
    assert isinstance(breaker.state, pybreaker.CircuitHalfOpenState)


def test_unknown_storage_state_raises_value_error():
    with pytest.raises(ValueError, match="Unknown state"):
        CircuitBreaker(state_storage=CircuitMemoryStorage("bogus"))


def test_memory_storage_defaults():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    assert storage.name == "memory"
    assert storage.state == STATE_CLOSED
    assert storage.counter == 0
    assert storage.success_counter == 0
    assert storage.opened_at is None


def test_memory_storage_counters_increment_and_reset():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    storage.increment_counter()
    storage.increment_counter()
    storage.increment_success_counter()
    assert storage.counter == 2
    assert storage.success_counter == 1
    storage.reset_counter()
    assert storage.counter == 0
    assert storage.success_counter == 1
    storage.reset_success_counter()
    assert storage.success_counter == 0


def test_memory_storage_state_and_opened_at_setters():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    storage.state = STATE_OPEN
    assert storage.state == STATE_OPEN
    moment = datetime(2024, 5, 1, 8, 30, tzinfo=timezone.utc)
    storage.opened_at = moment
    assert storage.opened_at == moment


def test_base_storage_name_and_default_methods_do_nothing():
    base = pybreaker.CircuitBreakerStorage("custom")
    assert base.name == "custom"
    assert base.increment_counter() is None
    assert base.reset_counter() is None
    assert base.increment_success_counter() is None
    assert base.reset_success_counter() is None


def test_excluded_exceptions_default_empty_and_add():
    breaker = CircuitBreaker()
    assert breaker.excluded_exceptions == ()
    breaker.add_excluded_exception(ValueError)
    assert breaker.excluded_exceptions == (ValueError,)
    assert isinstance(breaker.excluded_exceptions, tuple)


def test_add_multiple_and_remove_excluded_exceptions():
    breaker = CircuitBreaker(exclude=[KeyError])
    breaker.add_excluded_exceptions(ValueError, TypeError)
    assert breaker.excluded_exceptions == (KeyError, ValueError, TypeError)
    breaker.remove_excluded_exception(ValueError)
    assert breaker.excluded_exceptions == (KeyError, TypeError)


def test_remove_missing_excluded_exception_raises_value_error():
    breaker = CircuitBreaker()
    with pytest.raises(ValueError):
        breaker.remove_excluded_exception(ValueError)


def test_constructor_copies_exclude_and_listener_sequences():
    excluded = [ValueError]
    listener = CircuitBreakerListener()
    listeners = [listener]
    breaker = CircuitBreaker(exclude=excluded, listeners=listeners)
    excluded.append(KeyError)
    listeners.append(CircuitBreakerListener())
    assert breaker.excluded_exceptions == (ValueError,)
    assert breaker.listeners == (listener,)


def test_listeners_default_empty_and_add_remove():
    breaker = CircuitBreaker()
    assert breaker.listeners == ()
    listener = CircuitBreakerListener()
    breaker.add_listener(listener)
    assert breaker.listeners == (listener,)
    breaker.remove_listener(listener)
    assert breaker.listeners == ()


def test_add_listeners_registers_in_order():
    breaker = CircuitBreaker()
    first, second = CircuitBreakerListener(), CircuitBreakerListener()
    breaker.add_listeners(first, second)
    assert breaker.listeners == (first, second)


def test_remove_unknown_listener_raises_value_error():
    breaker = CircuitBreaker()
    with pytest.raises(ValueError):
        breaker.remove_listener(CircuitBreakerListener())


@pytest.mark.skipif(pybreaker.HAS_REDIS_SUPPORT, reason="redis is installed")
def test_redis_storage_requires_redis_dependency():
    with pytest.raises(ImportError):
        pybreaker.CircuitRedisStorage(STATE_CLOSED, object())
