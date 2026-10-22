from datetime import datetime

import pybreaker
import pytest


class Boom(Exception):
    pass


def boom():
    raise Boom("boom")


def ok():
    return "ok"


def test_memory_storage_initial_state():
    assert pybreaker.CircuitMemoryStorage("closed").state == "closed"
    assert pybreaker.CircuitMemoryStorage("half-open").state == "half-open"


def test_memory_storage_state_setter():
    storage = pybreaker.CircuitMemoryStorage("closed")
    storage.state = "open"
    assert storage.state == "open"


def test_memory_storage_counters_start_at_zero():
    storage = pybreaker.CircuitMemoryStorage("closed")
    assert storage.counter == 0
    assert storage.success_counter == 0


def test_memory_storage_increment_and_reset_failure_counter():
    storage = pybreaker.CircuitMemoryStorage("closed")
    storage.increment_counter()
    storage.increment_counter()
    storage.increment_counter()
    assert storage.counter == 3
    storage.reset_counter()
    assert storage.counter == 0


def test_memory_storage_increment_and_reset_success_counter():
    storage = pybreaker.CircuitMemoryStorage("closed")
    storage.increment_success_counter()
    storage.increment_success_counter()
    assert storage.success_counter == 2
    storage.reset_success_counter()
    assert storage.success_counter == 0


def test_memory_storage_counters_are_independent():
    storage = pybreaker.CircuitMemoryStorage("closed")
    storage.increment_counter()
    assert storage.success_counter == 0
    storage.increment_success_counter()
    storage.reset_counter()
    assert storage.success_counter == 1


def test_memory_storage_opened_at_defaults_to_none_and_can_be_set():
    storage = pybreaker.CircuitMemoryStorage("closed")
    assert storage.opened_at is None
    moment = datetime(2024, 5, 6, 7, 8, 9)
    storage.opened_at = moment
    assert storage.opened_at == moment


def test_memory_storages_are_independent_instances():
    a = pybreaker.CircuitMemoryStorage("closed")
    b = pybreaker.CircuitMemoryStorage("closed")
    a.increment_counter()
    assert b.counter == 0


def test_breaker_uses_provided_storage_for_failure_counter():
    storage = pybreaker.CircuitMemoryStorage("closed")
    cb = pybreaker.CircuitBreaker(fail_max=5, state_storage=storage)
    with pytest.raises(Boom):
        cb.call(boom)
    with pytest.raises(Boom):
        cb.call(boom)
    assert storage.counter == 2
    assert cb.fail_counter == 2


def test_breaker_open_is_reflected_in_storage():
    storage = pybreaker.CircuitMemoryStorage("closed")
    cb = pybreaker.CircuitBreaker(state_storage=storage)
    cb.open()
    assert storage.state == "open"
    assert storage.opened_at is not None


def test_breakers_sharing_storage_share_failure_count():
    storage = pybreaker.CircuitMemoryStorage("closed")
    first = pybreaker.CircuitBreaker(fail_max=5, state_storage=storage)
    second = pybreaker.CircuitBreaker(fail_max=5, state_storage=storage)
    with pytest.raises(Boom):
        first.call(boom)
    with pytest.raises(Boom):
        first.call(boom)
    assert second.fail_counter == 2


def test_breaker_starts_in_state_given_by_storage():
    storage = pybreaker.CircuitMemoryStorage("half-open")
    cb = pybreaker.CircuitBreaker(state_storage=storage)
    assert cb.current_state == "half-open"
