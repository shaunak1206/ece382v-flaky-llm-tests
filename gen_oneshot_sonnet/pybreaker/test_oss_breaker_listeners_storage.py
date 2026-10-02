"""Tests for CircuitBreakerListener callbacks, CircuitMemoryStorage and counters."""

import datetime

import pytest
from freezegun import freeze_time

import pybreaker
from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerListener,
    CircuitBreakerStorage,
    CircuitMemoryStorage,
)


class Boom(Exception):
    pass


class BusinessError(Exception):
    pass


def fail():
    raise Boom("boom")


def ok(*args, **kwargs):
    return "ok"


class Recorder(CircuitBreakerListener):
    def __init__(self):
        self.events = []
        self.breakers = []

    def before_call(self, cb, func, *args, **kwargs):
        self.breakers.append(cb)
        self.events.append(("before_call", func, args, kwargs))

    def failure(self, cb, exc):
        self.events.append(("failure", type(exc)))

    def success(self, cb):
        self.events.append(("success",))

    def state_change(self, cb, old_state, new_state):
        old = old_state.name if old_state is not None else None
        self.events.append(("state_change", old, new_state.name))

    @property
    def names(self):
        return [e[0] for e in self.events]

    @property
    def transitions(self):
        return [(e[1], e[2]) for e in self.events if e[0] == "state_change"]


@pytest.fixture
def frozen():
    with freeze_time("2024-01-01 12:00:00") as f:
        yield f


# ---------------------------------------------------------------- listeners


def test_base_listener_callbacks_are_noops():
    breaker = CircuitBreaker()
    listener = CircuitBreakerListener()
    assert listener.before_call(breaker, ok, 1, key=2) is None
    assert listener.failure(breaker, Boom()) is None
    assert listener.success(breaker) is None
    assert listener.state_change(breaker, None, breaker.state) is None


def test_listener_sees_before_call_and_success():
    rec = Recorder()
    breaker = CircuitBreaker(listeners=[rec])
    breaker.call(ok, 1, key=2)
    assert rec.names == ["before_call", "success"]
    assert rec.events[0] == ("before_call", ok, (1,), {"key": 2})
    assert rec.breakers == [breaker]


def test_listener_sees_failure():
    rec = Recorder()
    breaker = CircuitBreaker(fail_max=5, listeners=[rec])
    with pytest.raises(Boom):
        breaker.call(fail)
    assert rec.names == ["before_call", "failure"]
    assert rec.events[1] == ("failure", Boom)


def test_listener_sees_state_change_when_circuit_trips():
    rec = Recorder()
    breaker = CircuitBreaker(fail_max=1, listeners=[rec])
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert rec.names == ["before_call", "failure", "state_change"]
    assert rec.transitions == [(STATE_CLOSED, STATE_OPEN)]


def test_listener_sees_full_recovery_lifecycle(frozen):
    rec = Recorder()
    breaker = CircuitBreaker(fail_max=1, reset_timeout=10, listeners=[rec])
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)

    frozen.move_to("2024-01-01 12:00:11")
    assert breaker.call(ok) == "ok"
    assert rec.transitions == [
        (STATE_CLOSED, STATE_OPEN),
        (STATE_OPEN, STATE_HALF_OPEN),
        (STATE_HALF_OPEN, STATE_CLOSED),
    ]


def test_listener_sees_manual_state_changes():
    rec = Recorder()
    breaker = CircuitBreaker(listeners=[rec])
    breaker.open()
    breaker.half_open()
    breaker.close()
    assert rec.transitions == [
        (STATE_CLOSED, STATE_OPEN),
        (STATE_OPEN, STATE_HALF_OPEN),
        (STATE_HALF_OPEN, STATE_CLOSED),
    ]


def test_creating_a_breaker_does_not_notify_listeners():
    rec = Recorder()
    CircuitBreaker(listeners=[rec])
    assert rec.events == []


def test_excluded_exception_notifies_success_not_failure():
    rec = Recorder()
    breaker = CircuitBreaker(exclude=[BusinessError], listeners=[rec])

    def business():
        raise BusinessError("expected")

    with pytest.raises(BusinessError):
        breaker.call(business)
    assert rec.names == ["before_call", "success"]


def test_add_remove_and_list_listeners():
    a, b, c = Recorder(), Recorder(), Recorder()
    breaker = CircuitBreaker(listeners=[a])
    assert breaker.listeners == (a,)
    breaker.add_listener(b)
    breaker.add_listeners(c)
    assert breaker.listeners == (a, b, c)
    breaker.remove_listener(b)
    assert breaker.listeners == (a, c)
    with pytest.raises(ValueError):
        breaker.remove_listener(b)


def test_all_registered_listeners_are_notified_and_removed_ones_are_not():
    first, second = Recorder(), Recorder()
    breaker = CircuitBreaker(listeners=[first, second])
    breaker.call(ok)
    assert first.names == second.names == ["before_call", "success"]

    breaker.remove_listener(second)
    breaker.call(ok)
    assert len(first.names) == 4
    assert len(second.names) == 2


def test_listener_overriding_only_one_callback_works():
    failures = []

    class OnlyFailures(CircuitBreakerListener):
        def failure(self, cb, exc):
            failures.append(exc)

    breaker = CircuitBreaker(fail_max=5, listeners=[OnlyFailures()])
    breaker.call(ok)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert len(failures) == 1
    assert isinstance(failures[0], Boom)


# ------------------------------------------------------------------ storage


def test_memory_storage_initial_values():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    assert storage.name == "memory"
    assert storage.state == STATE_CLOSED
    assert storage.counter == 0
    assert storage.success_counter == 0
    assert storage.opened_at is None


def test_memory_storage_state_and_opened_at_setters():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    storage.state = STATE_OPEN
    assert storage.state == STATE_OPEN
    moment = datetime.datetime(2024, 5, 6, 7, 8, 9, tzinfo=datetime.timezone.utc)
    storage.opened_at = moment
    assert storage.opened_at == moment


def test_memory_storage_counters_increment_and_reset():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    for _ in range(3):
        storage.increment_counter()
    storage.increment_success_counter()
    assert (storage.counter, storage.success_counter) == (3, 1)
    storage.reset_counter()
    assert (storage.counter, storage.success_counter) == (0, 1)
    storage.reset_success_counter()
    assert storage.success_counter == 0


def test_breaker_counters_reflect_storage():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    breaker = CircuitBreaker(fail_max=5, state_storage=storage)
    for expected in (1, 2, 3):
        with pytest.raises(Boom):
            breaker.call(fail)
        assert breaker.fail_counter == expected == storage.counter


def test_opening_records_opened_at_in_storage(frozen):
    storage = CircuitMemoryStorage(STATE_CLOSED)
    breaker = CircuitBreaker(fail_max=1, state_storage=storage)
    assert storage.opened_at is None
    with pytest.raises(CircuitBreakerError):
        breaker.call(fail)
    assert storage.state == STATE_OPEN
    assert storage.opened_at == datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )


def test_breakers_sharing_storage_share_state():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    first = CircuitBreaker(fail_max=1, state_storage=storage)
    rec = Recorder()
    second = CircuitBreaker(fail_max=1, state_storage=storage, listeners=[rec])

    with pytest.raises(CircuitBreakerError):
        first.call(fail)
    assert second.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError, match="Timeout not elapsed yet"):
        second.call(ok)
    assert rec.transitions == [(STATE_CLOSED, STATE_OPEN)]


def test_breaker_can_start_from_half_open_storage():
    breaker = CircuitBreaker(state_storage=CircuitMemoryStorage(STATE_HALF_OPEN))
    assert breaker.current_state == STATE_HALF_OPEN
    assert breaker.call(ok) == "ok"
    assert breaker.current_state == STATE_CLOSED


def test_base_storage_exposes_name():
    assert CircuitBreakerStorage("custom").name == "custom"


@pytest.mark.skipif(pybreaker.HAS_REDIS_SUPPORT, reason="redis is installed")
def test_redis_storage_requires_redis_package():
    with pytest.raises(ImportError):
        pybreaker.CircuitRedisStorage(STATE_CLOSED, object())
