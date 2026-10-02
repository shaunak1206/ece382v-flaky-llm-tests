"""Tests for CircuitBreakerListener callbacks and call_async support."""
import time

import pytest
from tornado import gen

from pybreaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerListener,
)


class ServiceError(Exception):
    pass


def failing():
    raise ServiceError("boom")


def succeeding():
    return "ok"


class Recorder(CircuitBreakerListener):
    def __init__(self):
        self.before_calls = []
        self.failures = []
        self.successes = 0
        self.changes = []

    def before_call(self, cb, func, *args, **kwargs):
        self.before_calls.append((cb, func, args, kwargs))

    def failure(self, cb, exc):
        self.failures.append((cb, exc))

    def success(self, cb):
        self.successes += 1

    def state_change(self, cb, old_state, new_state):
        self.changes.append((old_state.name if old_state else None, new_state.name))


@pytest.fixture
def recorder():
    return Recorder()




def test_listener_before_call_receives_arguments(recorder):
    cb = CircuitBreaker(listeners=[recorder])
    cb.call(lambda a, b=1: a + b, 1, b=2)
    assert len(recorder.before_calls) == 1
    got_cb, _func, args, kwargs = recorder.before_calls[0]
    assert got_cb is cb
    assert args == (1,)
    assert kwargs == {"b": 2}


def test_listener_success_called_on_success(recorder):
    cb = CircuitBreaker(listeners=[recorder])
    cb.call(succeeding)
    cb.call(succeeding)
    assert recorder.successes == 2
    assert recorder.failures == []


def test_listener_failure_called_with_exception(recorder):
    cb = CircuitBreaker(fail_max=5, listeners=[recorder])
    with pytest.raises(ServiceError):
        cb.call(failing)
    assert len(recorder.failures) == 1
    got_cb, exc = recorder.failures[0]
    assert got_cb is cb
    assert isinstance(exc, ServiceError)
    assert recorder.successes == 0


def test_listener_state_change_on_trip(recorder):
    cb = CircuitBreaker(fail_max=2, reset_timeout=10, listeners=[recorder])
    with pytest.raises(ServiceError):
        cb.call(failing)
    assert recorder.changes == []
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert recorder.changes == [(STATE_CLOSED, STATE_OPEN)]


def test_listener_sees_full_recovery_cycle(recorder):
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, listeners=[recorder])
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)
    cb.call(succeeding)
    assert recorder.changes == [
        (STATE_CLOSED, STATE_OPEN),
        (STATE_OPEN, STATE_HALF_OPEN),
        (STATE_HALF_OPEN, STATE_CLOSED),
    ]


def test_listener_notified_on_failed_trial_call(recorder):
    cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, listeners=[recorder])
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    time.sleep(0.2)
    with pytest.raises(CircuitBreakerError):
        cb.call(failing)
    assert recorder.changes[-2:] == [(STATE_OPEN, STATE_HALF_OPEN), (STATE_HALF_OPEN, STATE_OPEN)]


def test_listener_state_change_for_manual_transitions(recorder):
    cb = CircuitBreaker(listeners=[recorder])
    cb.open()
    cb.half_open()
    cb.close()
    assert recorder.changes == [
        (STATE_CLOSED, STATE_OPEN),
        (STATE_OPEN, STATE_HALF_OPEN),
        (STATE_HALF_OPEN, STATE_CLOSED),
    ]




def test_excluded_exception_triggers_success_not_failure(recorder):
    cb = CircuitBreaker(exclude=[KeyError], listeners=[recorder])
    with pytest.raises(KeyError):
        cb.call(lambda: {}["x"])
    assert recorder.failures == []
    assert recorder.successes == 1


def test_add_listener(recorder):
    cb = CircuitBreaker()
    cb.add_listener(recorder)
    assert recorder in cb.listeners
    cb.call(succeeding)
    assert recorder.successes == 1




def test_remove_listener(recorder):
    cb = CircuitBreaker(listeners=[recorder])
    cb.remove_listener(recorder)
    assert cb.listeners == ()
    cb.call(succeeding)
    assert recorder.successes == 0








# ---- call_async -----------------------------------------------------------


@pytest.mark.asyncio
async def test_call_async_success():
    cb = CircuitBreaker(fail_max=2)

    async def work(x):
        return x * 2

    assert await cb.call_async(work, 21) == 42
    assert cb.fail_counter == 0


@pytest.mark.asyncio
async def test_call_async_with_tornado_coroutine():
    cb = CircuitBreaker(fail_max=2)

    @gen.coroutine
    def work():
        yield gen.moment
        raise gen.Return("tornado")

    assert await cb.call_async(work) == "tornado"


@pytest.mark.asyncio
async def test_call_async_failure_counts_and_reraises():
    cb = CircuitBreaker(fail_max=3)

    async def work():
        raise ServiceError()

    with pytest.raises(ServiceError):
        await cb.call_async(work)
    assert cb.fail_counter == 1


@pytest.mark.asyncio
async def test_call_async_opens_circuit():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    async def work():
        raise ServiceError()

    with pytest.raises(ServiceError):
        await cb.call_async(work)
    with pytest.raises(CircuitBreakerError):
        await cb.call_async(work)
    assert cb.current_state == STATE_OPEN
    with pytest.raises(CircuitBreakerError):
        await cb.call_async(work)


@pytest.mark.asyncio
async def test_call_async_half_open_success_closes():
    cb = CircuitBreaker(fail_max=1, reset_timeout=10)
    cb.half_open()

    async def work():
        return "recovered"

    assert await cb.call_async(work) == "recovered"
    assert cb.current_state == STATE_CLOSED


@pytest.mark.asyncio
async def test_call_async_half_open_failure_opens():
    cb = CircuitBreaker(fail_max=5, reset_timeout=10)
    cb.half_open()

    async def work():
        raise ServiceError()

    with pytest.raises(CircuitBreakerError):
        await cb.call_async(work)
    assert cb.current_state == STATE_OPEN


@pytest.mark.asyncio
async def test_call_async_notifies_listeners():
    rec = Recorder()
    cb = CircuitBreaker(fail_max=5, listeners=[rec])

    async def ok():
        return 1

    async def bad():
        raise ServiceError()

    await cb.call_async(ok)
    with pytest.raises(ServiceError):
        await cb.call_async(bad)
    assert len(rec.before_calls) == 2
    assert rec.successes == 1
    assert len(rec.failures) == 1


@pytest.mark.asyncio
async def test_call_async_excluded_exception():
    cb = CircuitBreaker(fail_max=1, exclude=[KeyError])

    async def work():
        raise KeyError("x")

    with pytest.raises(KeyError):
        await cb.call_async(work)
    assert cb.current_state == STATE_CLOSED


@pytest.mark.asyncio
async def test_async_decorator_via_call_async_flag():
    cb = CircuitBreaker(fail_max=2, reset_timeout=10)

    @cb(**{"__pybreaker_call_async": True})
    async def work(x):
        return x + 1

    assert await work(1) == 2


@pytest.mark.asyncio
async def test_async_decorator_trips_circuit():
    cb = CircuitBreaker(fail_max=1, reset_timeout=10)

    @cb(**{"__pybreaker_call_async": True})
    async def work():
        raise ServiceError()

    with pytest.raises(CircuitBreakerError):
        await work()
    assert cb.current_state == STATE_OPEN
