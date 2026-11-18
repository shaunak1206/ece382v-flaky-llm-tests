import pytest
from event_bus import EventBus


def test_eventbus_can_be_instantiated():
    bus = EventBus()
    assert isinstance(bus, EventBus)


def test_eventbus_has_subscribe_method():
    bus = EventBus()
    assert hasattr(bus, 'subscribe')
    assert callable(bus.subscribe)


def test_eventbus_has_publish_method():
    bus = EventBus()
    assert hasattr(bus, 'publish')
    assert callable(bus.publish)


def test_subscribe_registers_handler_and_publishes_calls_it():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('test_event', handler)
    bus.publish('test_event', 'payload')
    assert calls == ['payload']


def test_subscribe_registers_multiple_handlers_for_same_event():
    bus = EventBus()
    calls1 = []
    calls2 = []

    def handler1(data):
        calls1.append(data)

    def handler2(data):
        calls2.append(data)

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.publish('event', 'data')
    assert calls1 == ['data']
    assert calls2 == ['data']


def test_subscribe_register_multiple_events_handlers_are_independent():
    bus = EventBus()
    calls1 = []
    calls2 = []

    def handler1(data):
        calls1.append(data)

    def handler2(data):
        calls2.append(data)

    bus.subscribe('event1', handler1)
    bus.subscribe('event2', handler2)
    bus.publish('event1', 'data1')
    bus.publish('event2', 'data2')
    assert calls1 == ['data1']
    assert calls2 == ['data2']


def test_publishing_event_with_no_subscribers_does_not_error():
    bus = EventBus()
    # Should not raise any exception
    bus.publish('no_subscribers_event', 'something')


def test_handler_receives_event_data_correctly():
    bus = EventBus()

    class Payload:
        pass

    payload = Payload()

    captured = []

    def handler(data):
        captured.append(data)

    bus.subscribe('event', handler)
    bus.publish('event', payload)
    assert captured[0] is payload


def test_subscribe_raises_if_handler_is_not_callable():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe('event', 'not_callable')


def test_subscribe_multiple_handlers_are_called_in_order():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append('h1')

    def handler2(data):
        calls.append('h2')

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)
    bus.publish('event', None)
    assert calls == ['h1', 'h2']


def test_handlers_can_subscribe_other_events_during_execution():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append('first')
        bus.subscribe('event2', handler2)

    def handler2(data):
        calls.append('second')

    bus.subscribe('event1', handler1)
    bus.publish('event1', None)
    bus.publish('event2', None)
    assert calls == ['first', 'second']


def test_eventbus_version_string():
    from event_bus import __version__
    assert isinstance(__version__, str)
    assert __version__ == '1.0.0'


def test_multiple_publish_calls_call_handlers_multiple_times():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('repeat_event', handler)
    bus.publish('repeat_event', 1)
    bus.publish('repeat_event', 2)
    bus.publish('repeat_event', 3)
    assert calls == [1, 2, 3]


def test_handler_can_modify_external_state():
    bus = EventBus()
    state = {'count': 0}

    def handler(data):
        state['count'] += data

    bus.subscribe('inc', handler)
    bus.publish('inc', 1)
    bus.publish('inc', 2)
    assert state['count'] == 3


def test_subscribe_and_publish_with_none_event_name():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    # None as event name
    bus.subscribe(None, handler)
    bus.publish(None, 'data')
    assert calls == ['data']


def test_subscribe_same_handler_multiple_times():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe('event', handler)
    bus.subscribe('event', handler)
    bus.publish('event', 'data')
    # handler called twice
    assert calls == ['data', 'data']


def test_subscribe_handler_with_kwargs(capfd):
    bus = EventBus()
    result = {}

    def handler(data, flag=False):
        result['called'] = True
        result['flag'] = flag

    # We need to wrap handler to accept single argument,
    # since usually EventBus probably calls handlers with one argument (the data).
    # So check handler is called with data only and kwargs are ignored.

    def adapter(data):
        handler(data)

    bus.subscribe('event', adapter)
    bus.publish('event', 'payload')

    assert result.get('called')
    assert result.get('flag') is False


def test_handler_raises_inside_does_not_prevent_other_handlers():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append('h1')
        raise ValueError('fail')

    def handler2(data):
        calls.append('h2')

    bus.subscribe('event', handler1)
    bus.subscribe('event', handler2)

    # publishing still calls all handlers, exceptions from one don't propagate
    try:
        bus.publish('event', None)
    except ValueError:
        pytest.fail("Exception propagated from handler")

    assert calls == ['h1', 'h2']
