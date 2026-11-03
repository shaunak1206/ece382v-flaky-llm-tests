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

    bus.subscribe('test_event', handler)
    bus.emit('test_event', 'payload')

    assert received == ['payload']

def test_subscribe_multiple_handlers():
    bus = EventBus()
    results = []

    def handler1(data):
        results.append(('handler1', data))

    def handler2(data):
        results.append(('handler2', data))

    bus.subscribe('multi_event', handler1)
    bus.subscribe('multi_event', handler2)
    bus.emit('multi_event', 123)

    assert ('handler1', 123) in results
    assert ('handler2', 123) in results
    assert len(results) == 2

def test_emit_with_no_subscribers_does_not_fail():
    bus = EventBus()
    # Emitting event with no subscribers should not raise
    bus.emit('no_one_listens', 42)

def test_subscribe_same_handler_multiple_times_results_in_multiple_calls():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('dup_event', handler)
    bus.subscribe('dup_event', handler)
    bus.emit('dup_event', 'once')

    assert calls == ['once', 'once']

def test_unsubscribe_handler_prevents_future_calls():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('my_event', handler)
    bus.emit('my_event', 1)
    bus.unsubscribe('my_event', handler)
    bus.emit('my_event', 2)

    assert calls == [1]

def test_unsubscribe_handler_not_subscribed_does_not_raise():
    bus = EventBus()

    def handler(data):
        pass

    # Should not raise even if handler is not subscribed
    bus.unsubscribe('missing_event', handler)
    bus.subscribe('event', handler)
    bus.unsubscribe('event', lambda x: x)
    # no error expected

def test_subscribe_handler_for_multiple_events():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event1', handler)
    bus.subscribe('event2', handler)

    bus.emit('event1', 'a')
    bus.emit('event2', 'b')

    assert calls == ['a', 'b']

def test_handler_receives_correct_arguments():
    bus = EventBus()
    results = []

    def handler(*args, **kwargs):
        results.append((args, kwargs))

    bus.subscribe('args_event', handler)
    bus.emit('args_event', 1, 2, key='value')

    assert results == [((1, 2), {'key': 'value'})]

def test_unsubscribe_multiple_handlers():
    bus = EventBus()
    calls = []

    def h1(data): calls.append(('h1', data))
    def h2(data): calls.append(('h2', data))

    bus.subscribe('ev', h1)
    bus.subscribe('ev', h2)
    bus.emit('ev', 0)
    assert len(calls) == 2

    calls.clear()
    bus.unsubscribe('ev', h1)
    bus.emit('ev', 1)
    assert calls == [('h2', 1)]

    calls.clear()
    bus.unsubscribe('ev', h2)
    bus.emit('ev', 2)
    assert calls == []

def test_handlers_are_called_in_order_of_subscription():
    bus = EventBus()
    order = []

    def h1(_): order.append('first')
    def h2(_): order.append('second')
    def h3(_): order.append('third')

    bus.subscribe('ordered', h1)
    bus.subscribe('ordered', h2)
    bus.subscribe('ordered', h3)
    bus.emit('ordered', None)

    assert order == ['first', 'second', 'third']

def test_subscribe_and_unsubscribe_with_different_events():
    bus = EventBus()
    calls = []

    def h(data):
        calls.append(data)

    bus.subscribe('e1', h)
    bus.subscribe('e2', h)

    bus.emit('e1', 10)
    bus.emit('e2', 20)

    bus.unsubscribe('e1', h)
    bus.emit('e1', 30)
    bus.emit('e2', 40)

    assert calls == [10, 20, 40]

def test_emit_passes_all_arguments_to_handlers():
    bus = EventBus()
    caught = []

    def handler(a, b, c=None):
        caught.append((a, b, c))

    bus.subscribe('event', handler)
    bus.emit('event', 1, 2, c=3)

    assert caught == [(1, 2, 3)]

def test_unsubscribe_handler_only_once_if_subscribed_multiple_times():
    bus = EventBus()
    calls = []

    def h(x): calls.append(x)

    bus.subscribe('x', h)
    bus.subscribe('x', h)
    bus.unsubscribe('x', h)
    bus.emit('x', 5)

    # one subscription removed, one remains, so one call expected
    assert calls == [5]

def test_unsubscribe_all_handlers_and_emit():
    bus = EventBus()
    calls = []

    def h1(x): calls.append(('h1',x))
    def h2(x): calls.append(('h2',x))

    bus.subscribe('e', h1)
    bus.subscribe('e', h2)
    bus.unsubscribe('e', h1)
    bus.unsubscribe('e', h2)
    bus.emit('e', 100)

    assert calls == []

def test_handler_raising_exception_does_not_stop_other_handlers():
    bus = EventBus()
    calls = []

    def h1(data):
        calls.append('h1_start')
        raise ValueError("fail")

    def h2(data):
        calls.append('h2_called')

    bus.subscribe('evt', h1)
    bus.subscribe('evt', h2)

    # In normal EventBus, exceptions propagate. If not wrapped, this will raise.
    # We'll catch exception here to test h2 called.
    with pytest.raises(ValueError):
        bus.emit('evt', None)

    assert 'h1_start' in calls
    assert 'h2_called' in calls

def test_version_string_exists():
    import event_bus
    assert isinstance(event_bus.__version__, str)
    assert event_bus.__version__ == '1.0.0'

def test_all_exports_eventbus():
    import event_bus
    assert 'EventBus' in event_bus.__all__
    # __all__ contains only EventBus
    assert event_bus.__all__ == ['EventBus']
