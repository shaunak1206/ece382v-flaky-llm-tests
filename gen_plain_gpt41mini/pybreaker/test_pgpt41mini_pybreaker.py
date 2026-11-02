import pytest
import types
import sys
import threading
from datetime import datetime, timedelta, timezone
from freezegun import freeze_time
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
    CircuitBreakersState,
    CircuitOpenState,
    CircuitHalfOpenState,
    CircuitClosedState,
    CircuitBreakerListener,
    CircuitBreakerStorage,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_HALF_OPEN,
)
import time

UTC = timezone.utc


# Helpers

def always_fail():
    raise ValueError("Fail")


def always_pass():
    return "success"


def generator_success():
    yield "yield1"
    yield "yield2"


def generator_fail():
    yield "yield1"
    raise ValueError("Fail inside generator")


class DummyListener(CircuitBreakerListener):
    def __init__(self):
        self.events = []

    def before_call(self, cb, func, *args, **kwargs):
        self.events.append(("before_call", func.__name__))

    def failure(self, cb, exc):
        self.events.append(("failure", type(exc)))

    def success(self, cb):
        self.events.append("success")

    def state_change(self, cb, old_state, new_state):
        self.events.append(("state_change", old_state.name if old_state else None, new_state.name))


# Tests for CircuitMemoryStorage

def test_circuit_memory_storage_initial_state():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    assert storage.state == STATE_CLOSED
    assert storage.counter == 0
    assert storage.success_counter == 0
    assert storage.opened_at is None

def test_circuit_memory_storage_counters():
    storage = CircuitMemoryStorage(STATE_OPEN)
    storage.increment_counter()
    storage.increment_counter()
    assert storage.counter == 2
    storage.reset_counter()
    assert storage.counter == 0

    storage.increment_success_counter()
    storage.increment_success_counter()
    storage.increment_success_counter()
    assert storage.success_counter == 3
    storage.reset_success_counter()
    assert storage.success_counter == 0

def test_circuit_memory_storage_opened_at_set_and_get():
    storage = CircuitMemoryStorage(STATE_CLOSED)
    now = datetime.now(UTC)
    storage.opened_at = now
    assert storage.opened_at == now

# CircuitBreaker basic creation and attributes

def test_circuitbreaker_init_defaults():
    cb = CircuitBreaker()
    assert cb.fail_max == 5
    assert cb.reset_timeout == 60
    assert cb.success_threshold == 1
    assert cb.current_state == STATE_CLOSED
    assert cb.name is None
    assert isinstance(cb._state_storage, CircuitMemoryStorage)
    assert cb.excluded_exceptions == ()

def test_circuitbreaker_init_custom_values():
    cb = CircuitBreaker(
        fail_max=3,
        reset_timeout=30,
        success_threshold=2,
        exclude=[ValueError],
        listeners=[DummyListener()],
        name="testbreaker",
        throw_new_error_on_trip=False,
    )
    assert cb.fail_max == 3
    assert cb.reset_timeout == 30
    assert cb.success_threshold == 2
    assert ValueError in cb.excluded_exceptions
    assert isinstance(cb.listeners[0], DummyListener)
    assert cb.name == "testbreaker"
    assert cb._throw_new_error_on_trip is False

def test_property_setters():
    cb = CircuitBreaker()
    cb.fail_max = 10
    cb.reset_timeout = 20
    cb.success_threshold = 5
    cb.name = "newname"
    assert cb.fail_max == 10
    assert cb.reset_timeout == 20
    assert cb.success_threshold == 5
    assert cb.name == "newname"

# Test excluded exceptions management

def test_add_and_remove_excluded_exception():
    cb = CircuitBreaker()
    cb.add_excluded_exception(ValueError)
    assert ValueError in cb.excluded_exceptions

    # Add multiple
    cb.add_excluded_exceptions(KeyError, IndexError)
    assert KeyError in cb.excluded_exceptions
    assert IndexError in cb.excluded_exceptions

    cb.remove_excluded_exception(ValueError)
    assert ValueError not in cb.excluded_exceptions

def test_is_system_error_with_excluded_type():
    cb = CircuitBreaker(exclude=[ValueError])

    assert cb.is_system_error(Exception()) is True
    assert cb.is_system_error(ValueError()) is False

def test_is_system_error_with_excluded_callable():
    cb = CircuitBreaker(exclude=[lambda e: isinstance(e, ValueError)])

    assert cb.is_system_error(ValueError()) is False
    assert cb.is_system_error(KeyError()) is True

# Increment fail counter

def test_inc_counter_increments_storage_counter():
    cb = CircuitBreaker()
    initial = cb.fail_counter
    cb._inc_counter()
    assert cb.fail_counter == initial + 1

# Add and remove listeners

def test_add_and_remove_listener():
    cb = CircuitBreaker()
    listener = DummyListener()

    cb.add_listener(listener)
    assert listener in cb.listeners

    cb.remove_listener(listener)
    assert listener not in cb.listeners

def test_add_listeners_adds_all():
    cb = CircuitBreaker()
    l1 = DummyListener()
    l2 = DummyListener()
    cb.add_listeners(l1, l2)
    assert l1 in cb.listeners
    assert l2 in cb.listeners

def test_listeners_property_returns_tuple():
    cb = CircuitBreaker()
    l = DummyListener()
    cb.add_listener(l)
    assert isinstance(cb.listeners, tuple)
    assert l in cb.listeners

# State transition

def test_create_new_state_returns_correct_instance():
    cb = CircuitBreaker()
    closed = cb._create_new_state(STATE_CLOSED)
    open_state = cb._create_new_state(STATE_OPEN)
    half_open = cb._create_new_state(STATE_HALF_OPEN)

    from pybreaker import CircuitClosedState, CircuitOpenState, CircuitHalfOpenState

    assert isinstance(closed, CircuitClosedState)
    assert isinstance(open_state, CircuitOpenState)
    assert isinstance(half_open, CircuitHalfOpenState)

def test_create_new_state_invalid_state_raises():
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        cb._create_new_state("invalid_state")

def test_state_property_updates_cached_state():
    cb = CircuitBreaker()
    orig_state = cb.state
    assert orig_state.name == cb.current_state

    # Force underlying storage state to something else and check update side effect
    cb._state_storage.state = STATE_OPEN
    # Trigger property access and cause update
    state = cb.state
    assert state.name == STATE_OPEN

def test_state_setter_changes_state_and_notifies():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)

    prev = cb.state
    cb.state = STATE_OPEN
    new_state = cb.state
    assert new_state.name == STATE_OPEN
    assert listener.events[-1][0] == "state_change"
    assert listener.events[-1][1] == prev.name
    assert listener.events[-1][2] == STATE_OPEN

# Calling functions

def test_call_success():
    cb = CircuitBreaker()
    result = cb.call(always_pass)
    assert result == "success"

def test_call_failure_opens_circuit_after_fail_max():
    cb = CircuitBreaker(fail_max=2)
    # Cause 2 failures, 3rd failure triggers open
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(always_fail)
    # Third failure should raise CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        cb.call(always_fail)
    assert cb.current_state == STATE_OPEN

def test_call_with_excluded_exception_does_not_increment_fail():
    cb = CircuitBreaker(fail_max=1, exclude=[ValueError])
    # Raise excluded exception (ValueError)
    with pytest.raises(ValueError):
        cb.call(always_fail)
    assert cb.fail_counter == 0

def test_call_generator_success():
    cb = CircuitBreaker()
    gen = cb.call(lambda: generator_success())
    assert isinstance(gen, types.GeneratorType)

    # Exhaust generator
    with pytest.raises(StopIteration):
        next(gen)
        next(gen)
        next(gen)

def test_call_generator_fail_catches_error():
    cb = CircuitBreaker()
    gen = cb.call(lambda: generator_fail())
    # The first next yields
    first = next(gen)
    assert first == "yield1"
    # The second next raises error handled by breaker
    with pytest.raises(ValueError):
        next(gen)

# Using with-calling context manager

def test_calling_context_manager_success():
    cb = CircuitBreaker()
    with cb.calling():
        pass

def test_calling_context_manager_failure():
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        with cb.calling():
            raise ValueError()

# Open half_open close methods

def test_open_sets_state_and_opened_at():
    cb = CircuitBreaker()
    with freeze_time("2023-01-01 10:00:00"):
        cb.open()
        assert cb.current_state == STATE_OPEN
        assert cb._state_storage.opened_at == datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)

def test_half_open_sets_state():
    cb = CircuitBreaker()
    cb.half_open()
    assert cb.current_state == STATE_HALF_OPEN

def test_close_sets_state_and_resets_success_counter():
    cb = CircuitBreaker()
    cb._state_storage.increment_success_counter()
    assert cb.success_counter > 0
    cb.close()
    assert cb.current_state == STATE_CLOSED
    assert cb.success_counter == 0

# __call__ decorator wrapper

def test_call_wrapper_decorator_sync():
    cb = CircuitBreaker()

    @cb()
    def testfunc(x):
        return x + 1

    result = testfunc(4)
    assert result == 5

def test_call_wrapper_decorator_passes_args_kwargs():
    cb = CircuitBreaker()

    @cb()
    def testfunc(a, b=2):
        return a + b

    assert testfunc(3) == 5
    assert testfunc(3, b=3) == 6

def test_call_wrapper_decorator_with_call_async_false_raises_no_import_error(monkeypatch):
    cb = CircuitBreaker()

    with pytest.raises(ImportError):
        cb.__call__(__pybreaker_call_async=True)

# Test CircuitBreakerError exception

def test_circuitbreaker_error_is_exception():
    err = CircuitBreakerError("errormsg")
    assert isinstance(err, Exception)
    assert str(err) == "errormsg"

# Test CircuitBreakerState base methods

def test_circuitbreakerstate_handle_error_increments_counter_and_notifies():
    class TestListener(CircuitBreakerListener):
        def __init__(self):
            self.failed = False
        def failure(self, cb, exc):
            self.failed = True

    cb = CircuitBreaker()
    listener = TestListener()
    cb.add_listener(listener)

    state = CircuitClosedState(cb)
    initial_counter = cb.fail_counter
    exc = ValueError("fail")
    with pytest.raises(ValueError):
        state._handle_error(exc, reraise=True)
    assert cb.fail_counter == initial_counter + 1
    assert listener.failed is True

def test_circuitbreakerstate_handle_error_not_system_error_calls_success():
    cb = CircuitBreaker(exclude=[ValueError])
    called = []

    class TestListener(CircuitBreakerListener):
        def success(self, cb):
            called.append("success")
        def failure(self, cb, exc):
            called.append("failure")

    listener = TestListener()
    cb.add_listener(listener)
    state = CircuitClosedState(cb)

    exc = ValueError("excluded")
    state._handle_error(exc, reraise=False)
    assert "success" in called
    assert "failure" not in called

def test_circuitbreakerstate_handle_success_resets_counter_and_notifies():
    called = []

    class TestListener(CircuitBreakerListener):
        def success(self, cb):
            called.append("success")

    cb = CircuitBreaker()
    listener = TestListener()
    cb.add_listener(listener)
    state = CircuitClosedState(cb)
    state._breaker._state_storage.increment_counter()
    state._handle_success()
    assert state._breaker.fail_counter == 0
    assert "success" in called

def test_circuitbreakerstate_call_calls_listeners_and_executes_func():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)
    state = CircuitClosedState(cb)

    called = []

    def fun(x, y=0):
        called.append((x,y))
        return 42

    result = state.call(fun, 1, y=2)
    assert result == 42
    assert ("before_call", "fun") in listener.events
    assert ("before_call", "fun") not in listener.events  # already checked
    assert called == [(1, 2)]

def test_circuitbreakerstate_call_raises_and_handles_error():
    cb = CircuitBreaker()
    cb._fail_max = 1
    state = CircuitClosedState(cb)

    with pytest.raises(ValueError):
        state.call(always_fail)
    assert cb.current_state == STATE_OPEN

def test_circuitbreakerstate_call_async_runs(monkeypatch):
    # This test requires tornado support to run fully, test minimal coverage only
    cb = CircuitBreaker()
    state = CircuitClosedState(cb)
    called = []

    async_called = []

    # Emulate simple async coroutine function
    async def async_fun(x):
        async_called.append(x)
        return "ok"

    # Since call_async uses tornado.gen.coroutine, we cannot run fully here
    # Just check that the call_async returns coroutine function
    co = state.call_async(async_fun, 5)
    # co is a generator/coroutine object - test yield is successful
    # Can't test beyond without tornado running, so just check type
    assert hasattr(co, "__iter__") or hasattr(co, "__await__")

def test_circuitbreakerstate_generator_call_successful(monkeypatch):
    cb = CircuitBreaker()

    class State(CircuitBreakerState):
        def __init__(self, cb):
            super().__init__(cb, "test")
            self._breaker = cb

    state = State(cb)

    def genfun():
        yield "first"
        yield "second"

    gen = genfun()
    g = state.generator_call(gen)
    # Since generator_call itself is a generator, advance through it
    first = next(g)
    assert first == "first"
    second = g.send(None)
    assert second == "second"
    with pytest.raises(StopIteration):
        next(g)

def test_circuitbreakerstate_generator_call_fail(monkeypatch):
    cb = CircuitBreaker()
    state = CircuitClosedState(cb)

    def genfun():
        yield "first"
        raise ValueError("fail")

    gen = genfun()
    g = state.generator_call(gen)
    first = next(g)
    assert first == "first"
    with pytest.raises(ValueError):
        g.send(None)

# CircuitClosedState specific tests

def test_closedstate_on_failure_opens_circuit_and_raises_error():
    cb = CircuitBreaker(fail_max=1)
    state = CircuitClosedState(cb)
    cb._state_storage.increment_counter()
    with pytest.raises(CircuitBreakerError):
        state.on_failure(ValueError())

def test_closedstate_on_failure_not_above_threshold_raises_original():
    cb = CircuitBreaker(fail_max=10)
    state = CircuitClosedState(cb)
    cb._state_storage.increment_counter()
    with pytest.raises(ValueError):
        state.on_failure(ValueError())

# CircuitOpenState tests

def test_openstate_before_call_time_not_elapsed_raises_error():
    cb = CircuitBreaker(reset_timeout=60)
    cb.open()
    state = CircuitOpenState(cb)
    with freeze_time(datetime.now(UTC)):
        with pytest.raises(CircuitBreakerError):
            state.before_call(always_pass)

def test_openstate_before_call_time_elapsed_moves_to_half_open_and_calls():
    cb = CircuitBreaker(reset_timeout=1)
    cb.open()
    state = CircuitOpenState(cb)
    # Set opened_at in past to expire timeout
    past = datetime.now(UTC) - timedelta(seconds=2)
    cb._state_storage.opened_at = past

    called = []

    def fun():
        called.append(True)
        return 123

    result = state.before_call(fun)
    assert result == 123
    assert cb.current_state == STATE_HALF_OPEN

# CircuitHalfOpenState tests

def test_half_open_state_init_resets_success_counter_and_notifies():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)
    cb._state_storage._success_counter = 10
    old_state = CircuitClosedState(cb)
    state = CircuitHalfOpenState(cb, prev_state=old_state, notify=True)
    assert cb.success_counter == 0
    assert listener.events[-1][0] == "state_change"

def test_half_open_state_on_failure_opens_circuit_and_raises_error():
    cb = CircuitBreaker()
    cb._throw_new_error_on_trip = True
    state = CircuitHalfOpenState(cb, prev_state=None)
    with pytest.raises(CircuitBreakerError):
        state.on_failure(ValueError())

def test_half_open_state_on_failure_opens_circuit_and_raises_original_when_throw_new_error_false():
    cb = CircuitBreaker()
    cb._throw_new_error_on_trip = False
    state = CircuitHalfOpenState(cb, prev_state=None)
    with pytest.raises(ValueError):
        state.on_failure(ValueError())

def test_half_open_state_on_success_increments_and_closes():
    cb = CircuitBreaker(success_threshold=2)
    cb.half_open()
    storage = cb._state_storage
    storage.reset_success_counter()

    state = CircuitHalfOpenState(cb, prev_state=None)
    state._breaker = cb

    # Should not close circuit yet after 1 increment
    state.on_success()
    assert cb.current_state == STATE_HALF_OPEN
    assert cb.success_counter == 1

    # After second success it closes the circuit
    state.on_success()
    assert cb.current_state == STATE_CLOSED

def test_half_open_state_on_success_increments_success_counter():
    cb = CircuitBreaker(success_threshold=5)
    state = CircuitHalfOpenState(cb, prev_state=None)
    initial = cb.success_counter
    state.on_success()
    assert cb.success_counter == initial + 1

# CircuitBreaker call, call_async decorator

@pytest.mark.asyncio
async def test_call_async_decorator_runs(monkeypatch):
    try:
        import asyncio
        from tornado import gen
    except ImportError:
        pytest.skip("tornado or asyncio not available")

    cb = CircuitBreaker()

    called = []

    @cb(__pybreaker_call_async=True)
    @gen.coroutine
    def async_func(x):
        called.append(x)
        raise gen.Return(x + 1)

    result = await async_func(1)
    assert result == 2
    assert called == [1]

def test_open_returns_throw_new_error_on_trip():
    cb = CircuitBreaker()
    cb._throw_new_error_on_trip = True
    assert cb.open() is True
    cb._throw_new_error_on_trip = False
    assert cb.open() is False

# Excluded exception callable rules

def test_excluded_exceptions_callable_returns_false_for_excluded():
    cb = CircuitBreaker()

    def exclusion(exc):
        return isinstance(exc, KeyError)

    cb.add_excluded_exception(exclusion)

    assert cb.is_system_error(KeyError()) == False
    assert cb.is_system_error(ValueError()) == True

# Adding listeners concurrency test

def test_listener_threading_add_remove():
    cb = CircuitBreaker()
    l = DummyListener()

    def add_remove():
        cb.add_listener(l)
        cb.remove_listener(l)

    threads = [threading.Thread(target=add_remove) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Because add and remove done in thread, list should be empty
    assert l not in cb.listeners

def test_fail_max_setter_updates_value():
    cb = CircuitBreaker()
    cb.fail_max = 42
    assert cb.fail_max == 42

def test_reset_timeout_setter_updates_value():
    cb = CircuitBreaker()
    cb.reset_timeout = 177
    assert cb.reset_timeout == 177

def test_success_threshold_setter_updates_value():
    cb = CircuitBreaker()
    cb.success_threshold = 11
    assert cb.success_threshold == 11

def test_call_generator_call_catches_error_and_throws_into_generator():
    cb = CircuitBreaker()
    state = CircuitClosedState(cb)

    def gen_func():
        yield 1
        yield 2

    gen = gen_func()
    g = state.generator_call(gen)
    assert next(g) == 1

    # Send an error, it will call _handle_error with reraise=False and then call throw(e)
    with pytest.raises(ValueError):
        g.throw(ValueError)

def test_call_generator_call_handles_stopiteration_calls_success():
    cb = CircuitBreaker()
    cb._state_storage.increment_counter()
    state = CircuitClosedState(cb)

    def gen_func():
        yield 1
        return

    gen = gen_func()
    g = state.generator_call(gen)
    assert next(g) == 1
    with pytest.raises(StopIteration):
        next(g)
    # Success should reset fail counter
    assert cb.fail_counter == 0

def test_call_async_without_tornado_raises_import_error(monkeypatch):
    # Mimic no tornado support
    monkeypatch.setattr("pybreaker.HAS_TORNADO_SUPPORT", False)
    cb = CircuitBreaker()
    with pytest.raises(ImportError):
        cb.__call__(__pybreaker_call_async=True)

def test_circuitbreaker_name_setter_and_getter():
    cb = CircuitBreaker()
    cb.name = "foo"
    assert cb.name == "foo"

def test_circuitbreaker_state_change_notification():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)
    old_state = cb.state
    cb.state = STATE_OPEN
    assert listener.events[-1][0] == "state_change"
    assert listener.events[-1][1] == old_state.name
    assert listener.events[-1][2] == STATE_OPEN


# Test to ensure that in CircuitOpenState, when opened_at is None, it transitions to half-open

def test_openstate_before_call_opened_at_none_transitions_to_half_open(monkeypatch):
    cb = CircuitBreaker(reset_timeout=60)
    state = CircuitOpenState(cb)
    cb._state_storage._opened_at = None

    called = []

    def fun():
        called.append(True)
        return "hello"

    # no opened_at means timeout considered elapsed, should call and return result,
    # also transition state to half-open
    ret = state.before_call(fun)
    assert ret == "hello"
    assert cb.current_state == STATE_HALF_OPEN


# Test that calling CircuitBreaker.call with multiple args and kwargs passes them correctly to func

def test_call_passes_args_and_kwargs():
    cb = CircuitBreaker()

    def f(a, b, c=3):
        return a + b + c

    assert cb.call(f, 1, 2) == 6
    assert cb.call(f, 1, 2, c=4) == 7

# Test that CircuitBreaker.call raises CircuitBreakerError if state is open and timeout not elapsed
def test_call_raises_circuitbreakererror_if_state_open_timeout_not_elapsed():
    cb = CircuitBreaker(reset_timeout=60)
    cb.open()

    with pytest.raises(CircuitBreakerError):
        cb.call(always_pass)

# Test that CircuitBreaker.call can handle nested exceptions properly

def test_call_nested_exception_handling():
    cb = CircuitBreaker(fail_max=1)

    def func():
        def nested():
            raise ValueError("nested fail")
        nested()

    with pytest.raises(CircuitBreakerError):
        cb.call(func)

# Test that listeners before_call get called during call

def test_listener_before_call_called():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)

    cb.call(always_pass)
    # Check that before_call listener event is recorded
    assert any(event[0] == "before_call" for event in listener.events)

# Test that listeners success get called on success

def test_listener_success_called():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)

    cb.call(always_pass)
    assert "success" in listener.events

# Test that listeners failure get called on failure

def test_listener_failure_called():
    cb = CircuitBreaker(fail_max=1)
    listener = DummyListener()
    cb.add_listener(listener)
    with pytest.raises(ValueError):
        cb.call(always_fail)
    assert any(event[0] == "failure" for event in listener.events)

# Test half-open state increments success counter on success

def test_half_open_state_success_increments_success_counter(monkeypatch):
    cb = CircuitBreaker(success_threshold=2)
    cb.half_open()
    state = CircuitHalfOpenState(cb, prev_state=None)
    initial = cb.success_counter
    state.on_success()
    assert cb.success_counter == initial + 1

# Test half-open state closes if success_threshold reached

def test_half_open_state_closes_when_success_threshold_reached():
    cb = CircuitBreaker(success_threshold=1)
    cb.half_open()
    state = CircuitHalfOpenState(cb, prev_state=None)
    state.on_success()
    assert cb.current_state == STATE_CLOSED

# Test failure increments fail counter and opens circuit at threshold in closed state

def test_closed_state_failure_increments_fail_counter_and_opens(monkeypatch):
    cb = CircuitBreaker(fail_max=1)
    state = CircuitClosedState(cb)
    with pytest.raises(CircuitBreakerError):
        state._handle_error(ValueError(), reraise=True)
    assert cb.current_state == STATE_OPEN

# Test CircuitOpenState call delegates to before_call

def test_open_state_call_delegates_to_before_call(monkeypatch):
    cb = CircuitBreaker(reset_timeout=1)
    cb.open()
    state = CircuitOpenState(cb)
    past = datetime.now(UTC) - timedelta(seconds=2)
    cb._state_storage.opened_at = past

    def fun():
        return 5

    with freeze_time(datetime.now(UTC)):
        result = state.call(fun)
        assert result == 5
        assert cb.current_state == STATE_HALF_OPEN

# Test CircuitBreaker calling context manager can handle exceptions thrown inside

def test_calling_context_manager_raises_propagates():
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        with cb.calling():
            raise ValueError()

# Test adding duplicate excluded exceptions

def test_add_duplicate_excluded_exceptions():
    cb = CircuitBreaker()
    cb.add_excluded_exception(ValueError)
    cb.add_excluded_exception(ValueError)
    assert cb.excluded_exceptions.count(ValueError) == 2

# Exception types in excluded_exceptions are subclass checked

def test_is_system_error_with_subclass_exception():
    class MyExc(ValueError):
        pass

    cb = CircuitBreaker(exclude=[ValueError])
    assert cb.is_system_error(MyExc()) is False

# Test that CircuitOpenState does raise CircuitBreakerError if opened_at is in future (impossible, but test coverage)

def test_open_state_before_call_with_future_opened_at(monkeypatch):
    cb = CircuitBreaker(reset_timeout=60)
    now = datetime.now(UTC)
    cb._state_storage.opened_at = now + timedelta(seconds=1000)
    state = CircuitOpenState(cb)
    with pytest.raises(CircuitBreakerError):
        state.before_call(always_pass)

# Test CircuitMemoryStorage inherited methods raise TypeError if abstract methods not implemented (skipped since implemented)

# Test CircuitBreaker call with generator function returns generator

def test_call_generator_function_returns_generator_type():
    cb = CircuitBreaker()
    val = cb.call(generator_success)
    assert isinstance(val, types.GeneratorType)

# Test CircuitBreaker call with exception not system error calls success handler

def test_call_with_non_system_error_calls_success_handler():
    cb = CircuitBreaker(exclude=[ValueError])
    called = []

    class Listener(CircuitBreakerListener):
        def success(self, cb):
            called.append(True)
        def failure(self, cb, exc):
            called.append(False)

    listener = Listener()
    cb.add_listener(listener)
    with pytest.raises(ValueError):
        cb.call(always_fail)
    assert any(called)

# Test half_open_state on_failure raises CircuitBreakerError or original error depending on setting

def test_half_open_on_failure_raises_correct_error():
    cb = CircuitBreaker()
    cb._throw_new_error_on_trip = True
    state = CircuitHalfOpenState(cb, prev_state=None)
    with pytest.raises(CircuitBreakerError):
        state.on_failure(ValueError())
    cb._throw_new_error_on_trip = False
    with pytest.raises(ValueError):
        state.on_failure(ValueError())

# Test CircuitBreaker name getter and setter and default none

def test_circuitbreaker_name_property():
    cb = CircuitBreaker()
    assert cb.name is None
    cb.name = "newname"
    assert cb.name == "newname"

# Test open resets success counter

def test_open_resets_success_counter():
    cb = CircuitBreaker()
    cb._state_storage._success_counter = 10
    cb.open()
    assert cb.success_counter == 0

# Test half_open resets success counter

def test_half_open_resets_success_counter():
    cb = CircuitBreaker()
    cb._state_storage._success_counter = 10
    cb.half_open()
    assert cb.success_counter == 0

# Test close resets success counter

def test_close_resets_success_counter_again():
    cb = CircuitBreaker()
    cb._state_storage._success_counter = 10
    cb.close()
    assert cb.success_counter == 0

# Test that removing excluded exception not in list raises ValueError

def test_remove_excluded_exception_not_in_list_raises():
    cb = CircuitBreaker()
    with pytest.raises(ValueError):
        cb.remove_excluded_exception(ValueError)

# Test state setter with notify calls listener

def test_state_setter_notify_calls_listener():
    cb = CircuitBreaker()
    listener = DummyListener()
    cb.add_listener(listener)
    cb.state = STATE_CLOSED  # already closed but notify True in setter calls listener
    # Should have state_change event
    assert any(event[0] == "state_change" for event in listener.events)

# Test CircuitBreakerStorage abstract methods are present but raise if called directly

def test_circuitbreaker_storage_raises_on_abstract_methods():
    storage = CircuitBreakerStorage("abstract")

    with pytest.raises(TypeError):
        _ = storage.state

    with pytest.raises(TypeError):
        storage.state = STATE_CLOSED

    with pytest.raises(TypeError):
        storage.increment_counter()

    with pytest.raises(TypeError):
        storage.reset_counter()

    with pytest.raises(TypeError):
        storage.increment_success_counter()

    with pytest.raises(TypeError):
        storage.reset_success_counter()

    with pytest.raises(TypeError):
        _ = storage.counter

    with pytest.raises(TypeError):
        _ = storage.success_counter

    with pytest.raises(TypeError):
        _ = storage.opened_at

    with pytest.raises(TypeError):
        storage.opened_at = datetime.now(UTC)
