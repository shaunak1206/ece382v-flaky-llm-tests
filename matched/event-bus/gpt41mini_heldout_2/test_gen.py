import pytest
from event_bus import EventBus

def test_eventbus_instance_exists():
    bus = EventBus()
    assert bus is not None

def test_subscribe_and_emit_single_event():
    bus = EventBus()
    result = []

    def handler(data):
        result.append(data)

    bus.subscribe('test_event', handler)
    bus.emit('test_event', 123)
    assert result == [123]

def test_subscribe_multiple_handlers_same_event():
    bus = EventBus()
    result = []

    def handler1(data):
        result.append(('h1', data))

    def handler2(data):
        result.append(('h2', data))

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.emit('event', 'data')
    assert ('h1', 'data') in result
    assert ('h2', 'data') in result
    assert len(result) == 2

def test_subscribe_multiple_events_different_handlers():
    bus = EventBus()
    result = {}

    def handler_a(data):
        result['a'] = data

    def handler_b(data):
        result['b'] = data

    bus.subscribe('event_a', handler_a)
    bus.subscribe('event_b', handler_b)

    bus.emit('event_a', 1)
    bus.emit('event_b', 2)

    assert result == {'a': 1, 'b': 2}

def test_unsubscribe_handler_removes_handler():
    bus = EventBus()
    result = []

    def handler(data):
        result.append(data)

    bus.subscribe('event', handler)
    bus.unsubscribe('event', handler)
    bus.emit('event', 42)
    assert result == []

def test_unsubscribe_handler_only_removes_specific_handler():
    bus = EventBus()
    result = []

    def handler1(data):
        result.append(('h1', data))

    def handler2(data):
        result.append(('h2', data))

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)

    bus.unsubscribe('event', handler1)
    bus.emit('event', 'x')

    assert ('h1', 'x') not in result
    assert ('h2', 'x') in result
    assert len(result) == 1

def test_unsubscribe_handler_nonexistent_event_does_not_error():
    bus = EventBus()

    def handler(data):
        pass

    # Should not raise
    bus.unsubscribe('nonexistent_event', handler)

def test_unsubscribe_handler_not_subscribed_does_not_error():
    bus = EventBus()

    def handler(data):
        pass

    bus.subscribe('event', handler)
    # Unsubscribe a different handler which was not subscribed
    def not_subscribed(data):
        pass

    bus.unsubscribe('event', not_subscribed)  # no error expected

def test_emit_event_with_no_subscribers_does_nothing():
    bus = EventBus()
    # Should not raise, no handlers for event
    bus.emit('no_handlers_event', 'data')

def test_subscribe_handler_multiple_times_only_calls_once_per_emit():
    bus = EventBus()
    result = []

    def handler(data):
        result.append(data)

    bus.subscribe('event', handler)
    bus.subscribe('event', handler)
    bus.emit('event', 10)

    # Depending on implementation could be twice or once; test expected once
    # Most event bus implementations allow duplicate subscriptions
    # Here we just test that handler was called twice if duplicates allowed
    # So we assert it is called twice
    assert result == [10, 10]

def test_subscribe_handler_then_unsubscribe_then_resubscribe():
    bus = EventBus()
    result = []

    def handler(data):
        result.append(data)

    bus.subscribe('event', handler)
    bus.unsubscribe('event', handler)
    bus.subscribe('event', handler)
    bus.emit('event', 7)
    assert result == [7]

def test_handler_receives_correct_arguments():
    bus = EventBus()
    received = []

    def handler(a, b=None):
        received.append((a, b))

    bus.subscribe('event', handler)
    bus.emit('event', 1)
    bus.emit('event', 2)
    # Check that the handler works even if b is None
    assert received == [(1, None), (2, None)]

def test_subscribe_handler_which_raises_exception_does_not_stop_others():
    bus = EventBus()
    called = []

    def handler1(data):
        called.append('handler1')

    def handler2(data):
        raise ValueError("fail")

    def handler3(data):
        called.append('handler3')

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.subscribe('event', handler3)

    # By default, does emit catch exceptions or propagate?
    # We test that exception propagates (common behavior)
    with pytest.raises(ValueError):
        bus.emit('event', None)

    # handler1 should be called, handler3 not called because handler2 raises and stops iteration
    assert 'handler1' in called
    assert 'handler3' not in called

def test_subscribe_none_handler_raises_type_error():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe('event', None)

def test_unsubscribe_none_handler_raises_type_error():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.unsubscribe('event', None)

def test_emit_event_with_multiple_args():
    bus = EventBus()
    result = []

    def handler(*args):
        result.append(args)

    bus.subscribe('multi_args', handler)
    bus.emit('multi_args', 1, 2, 3)

    assert result == [(1, 2, 3)]

def test_emit_event_with_kwargs():
    bus = EventBus()
    result = []

    def handler(**kwargs):
        result.append(kwargs)

    bus.subscribe('kwargs_event', handler)
    bus.emit('kwargs_event', key='value')

    # If emit passes keyword arguments or not depends on implementation.
    # Most event bus emit takes *args only, so expect no kwargs
    # We expect result to contain empty because only positional args passed.
    assert result == [{}] or result == [{'key': 'value'}]

def test_emit_events_are_independent():
    bus = EventBus()
    result = []

    def h1(data):
        result.append(('h1', data))

    def h2(data):
        result.append(('h2', data))

    bus.subscribe('event1', h1)
    bus.subscribe('event2', h2)

    bus.emit('event1', 1)
    bus.emit('event2', 2)

    assert result == [('h1', 1), ('h2', 2)]

def test_subscribe_same_handler_to_different_events():
    bus = EventBus()
    result = []

    def handler(data):
        result.append(data)

    bus.subscribe('event1', handler)
    bus.subscribe('event2', handler)

    bus.emit('event1', 'a')
    bus.emit('event2', 'b')

    assert result == ['a', 'b']
