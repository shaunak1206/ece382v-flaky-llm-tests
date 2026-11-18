import pytest
from event_bus.bus import EventBus

def test_eventbus_can_be_instantiated():
    bus = EventBus()
    assert isinstance(bus, EventBus)

def test_subscribe_and_emit_calls_handler():
    bus = EventBus()
    called = []

    def handler(data):
        called.append(data)

    bus.subscribe('test_event', handler)
    bus.emit('test_event', 123)
    assert called == [123]

def test_subscribe_multiple_handlers():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.emit('event', 'data')
    assert ('h1', 'data') in calls
    assert ('h2', 'data') in calls
    assert len(calls) == 2

def test_unsubscribe_handler():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event', handler)
    bus.unsubscribe('event', handler)
    bus.emit('event', 'test')
    assert calls == []

def test_unsubscribe_nonexistent_handler_does_not_raise():
    bus = EventBus()

    def handler(data):
        pass

    bus.unsubscribe('nonexistent_event', handler)  # Should not raise

def test_emit_without_subscribers_does_not_fail():
    bus = EventBus()
    # No subscribers to this event, should not error or do anything
    bus.emit('no_listeners', 42)

def test_handler_receives_multiple_emits():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('repeat', handler)
    for i in range(3):
        bus.emit('repeat', i)
    assert calls == [0, 1, 2]

def test_subscribe_same_handler_multiple_times():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('dup_event', handler)
    bus.subscribe('dup_event', handler)  # Subscribe twice
    bus.emit('dup_event', 'a')
    # Depending on implementation, might be called once or twice. Typically once.
    # So test that it is called once
    assert calls == ['a']

def test_unsubscribe_handler_only_removes_that_handler():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.unsubscribe('event', handler1)
    bus.emit('event', 100)
    assert calls == [('h2', 100)]

def test_subscribe_handler_to_multiple_events():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event1', handler)
    bus.subscribe('event2', handler)
    bus.emit('event1', 1)
    bus.emit('event2', 2)
    assert calls == [1, 2]

def test_handlers_are_independent_across_events():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('event1', handler1)
    bus.subscribe('event2', handler2)
    bus.emit('event1', 'a')
    bus.emit('event2', 'b')
    assert calls == [('h1', 'a'), ('h2', 'b')]

def test_emit_passes_multiple_args_if_supported():
    bus = EventBus()
    calls = []

    # Check if handler supports varargs, assume implementation forwards all args
    def handler(*args):
        calls.append(args)

    bus.subscribe('multi_args', handler)
    bus.emit('multi_args', 1)
    bus.emit('multi_args', 1, 2)
    bus.emit('multi_args', 1, 2, 3)
    assert calls == [(1,), (1, 2), (1, 2, 3)]

def test_emit_passes_kwargs_if_supported():
    bus = EventBus()
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)

    bus.subscribe('kwargs_event', handler)
    bus.emit('kwargs_event', key=1)
    bus.emit('kwargs_event', key=1, other=2)
    assert calls == [{'key': 1}, {'key': 1, 'other': 2}]

def test_emit_passes_both_args_and_kwargs():
    bus = EventBus()
    calls = []

    def handler(*args, **kwargs):
        calls.append((args, kwargs))

    bus.subscribe('args_kwargs', handler)
    bus.emit('args_kwargs', 1, 2, key=3)
    assert calls == [((1, 2), {'key': 3})]

def test_subscribe_after_emit_still_receives_events():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.emit('new_event', 1)
    bus.subscribe('new_event', handler)
    bus.emit('new_event', 2)
    assert calls == [2]

def test_unsubscribe_all_handlers_one_by_one():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('ev', handler1)
    bus.subscribe('ev', handler2)
    bus.unsubscribe('ev', handler1)
    bus.emit('ev', 'a')
    bus.unsubscribe('ev', handler2)
    bus.emit('ev', 'b')
    assert calls == [('h2', 'a')]

def test_subscribe_and_unsubscribe_non_callable_raises():
    bus = EventBus()
    non_callable = 42
    with pytest.raises(TypeError):
        bus.subscribe('event', non_callable)
    with pytest.raises(TypeError):
        bus.unsubscribe('event', non_callable)

def test_handlers_called_in_order():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append('h1')

    def handler2(data):
        calls.append('h2')

    bus.subscribe('eo', handler1)
    bus.subscribe('eo', handler2)
    bus.emit('eo', None)
    assert calls == ['h1', 'h2']
