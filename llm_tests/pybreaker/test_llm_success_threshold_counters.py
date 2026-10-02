"""Tests for success_threshold, fail_counter, success_counter, and CircuitMemoryStorage."""
import time
from datetime import datetime, timezone

import pytest

from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerStorage,
    CircuitMemoryStorage,
)


class ServiceError(Exception):
    pass


def failing():
    raise ServiceError("boom")


def succeeding():
    return "ok"


def test_success_threshold_requires_multiple_successes():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=3)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)

    cb.call(succeeding)
    assert cb.current_state == STATE_HALF_OPEN
    assert cb.success_counter == 1
    cb.call(succeeding)
    assert cb.current_state == STATE_HALF_OPEN
    assert cb.success_counter == 2
    cb.call(succeeding)
    assert cb.current_state == STATE_CLOSED


def test_success_threshold_failure_in_half_open_reopens():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=2)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)
    cb.call(succeeding)
    assert cb.current_state == STATE_HALF_OPEN
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert cb.current_state == STATE_OPEN
    assert cb.success_counter == 0


def test_success_counter_reset_after_closing():
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=2)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)
    cb.call(succeeding)
    cb.call(succeeding)
    assert cb.current_state == STATE_CLOSED
    assert cb.success_counter == 0




def test_success_counter_resets_when_opened():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10, success_threshold=5)
    cb.half_open()
    cb.call(succeeding)
    assert cb.success_counter == 1
    cb.open()
    assert cb.success_counter == 0


def test_success_counter_resets_on_entering_half_open():
    cb = CircuitBreaker(success_threshold=5)
    cb.half_open()
    cb.call(succeeding)
    cb.call(succeeding)
    assert cb.success_counter == 2
    cb.half_open()
    assert cb.success_counter == 0




def test_success_threshold_setter_affects_closing():
    cb = CircuitBreaker(success_threshold=1)
    cb.success_threshold = 2
    cb.half_open()
    cb.call(succeeding)
    assert cb.current_state == STATE_HALF_OPEN
    cb.call(succeeding)
    assert cb.current_state == STATE_CLOSED


@pytest.mark.parametrize("failures", [1, 2, 3, 4])
def test_fail_counter_tracks_consecutive_failures(failures):
    cb = CircuitBreaker(fail_max=10)
    for _ in range(failures):
        with pytest.raises(ServiceError):
            cb.call(failing)
    assert cb.fail_counter == failures


@pytest.mark.parametrize("fail_max", [1, 2, 3, 5])
def test_breaker_opens_exactly_at_fail_max(fail_max):
    cb = CircuitBreaker(fail_max=fail_max, reset_timeout=10)
    for _ in range(fail_max - 1):
        with pytest.raises(ServiceError):
            cb.call(failing)
        assert cb.current_state == STATE_CLOSED
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert cb.current_state == STATE_OPEN


def test_fail_counter_resets_after_recovery():
    cb = CircuitBreaker(fail_max=2, reset_timeout=0.1)
    with pytest.raises(ServiceError):
        cb.call(failing)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert cb.fail_counter == 2
    time.sleep(0.2)
    cb.call(succeeding)
    assert cb.fail_counter == 0




# ---- CircuitMemoryStorage -------------------------------------------------


def test_memory_storage_initial_values():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    assert storage.name == "memory"
    assert storage.state == STATE_CLOSED
    assert storage.counter == 0
    assert storage.success_counter == 0
    assert storage.opened_at is None




def test_memory_storage_counters():
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




def test_memory_storage_opened_at_roundtrip():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    now = datetime.now(timezone.utc)
    storage.opened_at = now
    assert storage.opened_at == now


def test_breaker_open_records_opened_at():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    cb = CircuitBreaker(state_storage=storage)
    before = datetime.now(timezone.utc)
    cb.open()
    after = datetime.now(timezone.utc)
    assert before <= storage.opened_at <= after


def test_breaker_uses_provided_storage_for_counters():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    cb = CircuitBreaker(fail_max=5, state_storage=storage)
    with pytest.raises(ServiceError):
        cb.call(failing)
    assert storage.counter == 1
    assert cb.fail_counter == 1


def test_breaker_state_follows_shared_storage():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    cb1 = CircuitBreaker(fail_max=1, reset_timeout=10, state_storage=storage)
    cb2 = CircuitBreaker(fail_max=1, reset_timeout=10, state_storage=storage)
    with pytest.raises(CircuitBreakerError):
        cb1.call(failing)
    assert cb2.current_state == STATE_OPEN
    assert cb2.state.name == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        cb2.call(succeeding)


def test_open_storage_without_opened_at_allows_trial_call():
    cb = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_OPEN))
    assert cb.call(succeeding) == "ok"
    assert cb.current_state == STATE_CLOSED


