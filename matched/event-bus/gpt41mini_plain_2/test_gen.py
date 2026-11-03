import pytest
from event_bus.bus import EventBus


def test_eventbus_can_be_instantiated():
    bus = EventBus()
    assert isinstance(bus, EventBus)


def test_subscribe_and_emit_single_event():
    bus = EventBus()
    received = []

    def handler(data):
        received.append(data)

    bus.subscribe('test.event', handler)
    bus.emit('test.event', 123)

    assert received == [123]


def test_emit_without_subscribers_does_nothing():
    bus = EventBus()
    # Should not raise
    bus.emit('nonexistent.event', 'data')


def test_multiple_subscribers_receive_event():
    bus = EventBus()
    received1 = []
    received2 = []

    def handler1(data):
        received1.append(data)

    def handler2(data):
        received2.append(data)

    bus.subscribe('my.event', handler1)
    bus.subscribe('my.event', handler2)

    bus.emit('my.event', 'payload')

    assert received1 == ['payload']
    assert received2 == ['payload']


def test_subscribe_same_handler_multiple_times():
    bus = EventBus()
    called = []

    def handler(data):
        called.append(data)

    bus.subscribe('dup.event', handler)
    bus.subscribe('dup.event', handler)

    bus.emit('dup.event', 1)

    # It depends on implementation if multiple subscriptions add multiple calls
    # This test assumes each subscription adds an entry
    assert called == [1, 1]


def test_unsubscribe_removes_handler():
    bus = EventBus()
    called = []

    def handler(data):
        called.append(data)

    bus.subscribe('to.remove', handler)
    bus.unsubscribe('to.remove', handler)
    bus.emit('to.remove', 42)

    assert called == []


def test_unsubscribe_nonexistent_handler_does_not_fail():
    bus = EventBus()

    def handler(data):
        pass

    # Should not raise error
    bus.unsubscribe('no.event', handler)


def test_handler_receives_correct_data_types():
    bus = EventBus()
    received = []

    def handler(data):
        received.append(data)

    bus.subscribe('data.types', handler)
    bus.emit('data.types', [1, 2, 3])
    bus.emit('data.types', {'a': 1})
    bus.emit('data.types', None)

    assert received == [[1, 2, 3], {'a': 1}, None]


def test_handlers_receive_events_separately():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('x.event', handler1)
    bus.subscribe('y.event', handler2)

    bus.emit('x.event', 'foo')
    bus.emit('y.event', 'bar')

    assert calls == [('h1', 'foo'), ('h2', 'bar')]


def test_handlers_can_subscribe_during_emit():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))
        # Subscribe handler2 when first event is received
        bus.subscribe('dynamic.event', handler2)

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('dynamic.event', handler1)
    bus.emit('dynamic.event', 1)
    bus.emit('dynamic.event', 2)

    assert calls == [('h1', 1), ('h1', 2), ('h2', 2)]


def test_handlers_can_unsubscribe_during_emit():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))
        bus.unsubscribe('dynamic.remove', handler2)

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('dynamic.remove', handler1)
    bus.subscribe('dynamic.remove', handler2)

    bus.emit('dynamic.remove', 'a')
    bus.emit('dynamic.remove', 'b')

    # handler2 should be called only on first emit, then unsubscribed
    assert calls == [('h1', 'a'), ('h2', 'a'), ('h1', 'b')]


def test_subscribe_with_non_callable_fails():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe('event', None)


def test_unsubscribe_with_non_callable_fails():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.unsubscribe('event', None)


def test_emit_passes_multiple_arguments_to_handlers():
    bus = EventBus()
    calls = []

    def handler(*args, **kwargs):
        calls.append((args, kwargs))

    bus.subscribe('multi.args', handler)
    bus.emit('multi.args', 1, 2, key='value')

    assert calls == [((1, 2), {'key': 'value'})]


def test_subscribe_unsubscribe_multiple_handlers():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(('h1', data))

    def handler2(data):
        calls.append(('h2', data))

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.unsubscribe('event', handler1)

    bus.emit('event', 'payload')

    assert calls == [('h2', 'payload')]


def test_subscribe_multiple_events_with_same_handler():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event1', handler)
    bus.subscribe('event2', handler)

    bus.emit('event1', 1)
    bus.emit('event2', 2)

    assert calls == [1, 2]


def test_emit_event_with_no_data():
    bus = EventBus()
    calls = []

    def handler(data=None):
        calls.append(data)

    bus.subscribe('no.data', handler)
    bus.emit('no.data')

    assert calls == [None]


def test_unsubscribe_handler_only_from_specific_event():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event1', handler)
    bus.subscribe('event2', handler)

    bus.unsubscribe('event1', handler)
    bus.emit('event1', 'a')
    bus.emit('event2', 'b')

    assert calls == ['b']


def test_subscribe_with_different_events_stores_separately():
    bus = EventBus()
    bus.subscribe('event.a', lambda x: x)
    bus.subscribe('event.b', lambda x: x)

    # Internals might be private, but test no cross contamination by subscribing again
    # This test just checks no exceptions on multiple events subscription
    bus.emit('event.a', 1)
    bus.emit('event.b', 2)


def test_unsubscribe_nonexistent_event_handler_does_not_raise():
    bus = EventBus()

    def handler(data):
        pass

    # Unsubscribe from event with no handlers
    bus.unsubscribe('event.unknown', handler)
