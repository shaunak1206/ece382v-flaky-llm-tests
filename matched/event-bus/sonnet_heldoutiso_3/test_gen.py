import time

import pytest

import event_bus
from event_bus import EventBus


def _wait_for(predicate, timeout=2.0):
    """Wait until predicate is true (tolerates handlers running asynchronously)."""
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _settle(delay=0.1):
    time.sleep(delay)


@pytest.fixture
def bus():
    return EventBus()


def test_package_exports():
    assert event_bus.__version__ == '1.0.0'
    assert 'EventBus' in event_bus.__all__
    assert event_bus.EventBus is EventBus


def test_can_instantiate():
    assert EventBus() is not None


def test_on_decorator_registers_handler_and_emit_calls_it(bus):
    calls = []

    @bus.on('greet')
    def handler():
        calls.append(1)

    bus.emit('greet')
    assert _wait_for(lambda: len(calls) == 1)


def test_emit_passes_positional_args(bus):
    received = []

    @bus.on('data')
    def handler(a, b):
        received.append((a, b))

    bus.emit('data', 1, 2)
    assert _wait_for(lambda: received == [(1, 2)])


def test_emit_passes_keyword_args(bus):
    received = []

    @bus.on('kw')
    def handler(x=None, y=None):
        received.append((x, y))

    bus.emit('kw', x='a', y='b')
    assert _wait_for(lambda: received == [('a', 'b')])


def test_emit_passes_mixed_args_and_kwargs(bus):
    received = []

    @bus.on('mixed')
    def handler(a, b=None):
        received.append((a, b))

    bus.emit('mixed', 10, b=20)
    assert _wait_for(lambda: received == [(10, 20)])


def test_multiple_handlers_for_same_event_all_called(bus):
    calls = []

    @bus.on('multi')
    def first():
        calls.append('first')

    @bus.on('multi')
    def second():
        calls.append('second')

    bus.emit('multi')
    assert _wait_for(lambda: sorted(calls) == ['first', 'second'])


def test_handlers_only_called_for_their_event(bus):
    a_calls = []
    b_calls = []

    @bus.on('a')
    def on_a():
        a_calls.append(1)

    @bus.on('b')
    def on_b():
        b_calls.append(1)

    bus.emit('a')
    assert _wait_for(lambda: len(a_calls) == 1)
    _settle()
    assert b_calls == []


def test_handler_called_once_per_emit(bus):
    calls = []

    @bus.on('tick')
    def handler():
        calls.append(1)

    bus.emit('tick')
    bus.emit('tick')
    bus.emit('tick')
    assert _wait_for(lambda: len(calls) == 3)
    _settle()
    assert len(calls) == 3


def test_same_handler_registered_for_multiple_events(bus):
    calls = []

    def handler(value):
        calls.append(value)

    bus.on('one')(handler)
    bus.on('two')(handler)

    bus.emit('one', 1)
    bus.emit('two', 2)
    assert _wait_for(lambda: sorted(calls) == [1, 2])


def test_instances_are_independent():
    bus1 = EventBus()
    bus2 = EventBus()
    calls = []

    @bus1.on('evt')
    def handler():
        calls.append(1)

    bus2.emit('evt')
    _settle()
    assert calls == []

    bus1.emit('evt')
    assert _wait_for(lambda: len(calls) == 1)


def test_emit_with_no_handlers_does_not_call_other_handlers(bus):
    calls = []

    @bus.on('registered')
    def handler():
        calls.append(1)

    try:
        bus.emit('unregistered')
    except Exception:
        # Behaviour for unknown events is implementation defined;
        # the important thing is that unrelated handlers are not called.
        pass
    _settle()
    assert calls == []


def test_add_event_registers_handler(bus):
    if not hasattr(bus, 'add_event'):
        pytest.skip('add_event not available')
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, 'manual')
    bus.emit('manual')
    assert _wait_for(lambda: len(calls) == 1)


def test_remove_event_unregisters_handler(bus):
    if not hasattr(bus, 'add_event') or not hasattr(bus, 'remove_event'):
        pytest.skip('add_event/remove_event not available')
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, 'removable')
    bus.remove_event(handler.__name__, 'removable')
    try:
        bus.emit('removable')
    except Exception:
        pass
    _settle()
    assert calls == []
