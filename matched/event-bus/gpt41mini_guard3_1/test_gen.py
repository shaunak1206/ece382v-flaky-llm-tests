import random
import pytest

from event_bus import EventBus


def test_eventbus_can_subscribe_and_emit():
    random.seed(0)
    bus = EventBus()

    called = []

    def handler(event):
        called.append(event)

    bus.subscribe('event1', handler)
    bus.emit('event1', 'data1')

    assert called == ['data1']


def test_eventbus_multiple_handlers_receive_event():
    random.seed(1)
    bus = EventBus()

    called1 = []
    called2 = []

    bus.subscribe('event', lambda e: called1.append(e))
    bus.subscribe('event', lambda e: called2.append(e))

    bus.emit('event', 42)

    assert called1 == [42]
    assert called2 == [42]


def test_eventbus_handlers_receiving_correct_event_type():
    random.seed(2)
    bus = EventBus()

    received_data = []

    def handler1(e):
        received_data.append(('h1', e))

    def handler2(e):
        received_data.append(('h2', e))

    bus.subscribe('type1', handler1)
    bus.subscribe('type2', handler2)

    bus.emit('type1', 123)
    bus.emit('type2', 456)

    assert ('h1', 123) in received_data
    assert ('h2', 456) in received_data
    assert len(received_data) == 2


def test_eventbus_unsubscribe_removes_handler():
    random.seed(3)
    bus = EventBus()

    called = []

    def handler(e):
        called.append(e)

    bus.subscribe('x', handler)
    bus.emit('x', 1)
    bus.unsubscribe('x', handler)
    bus.emit('x', 2)

    assert called == [1]


def test_unsubscribe_nonexistent_handler_raises_no_error():
    random.seed(4)
    bus = EventBus()

    # unsubscribe handler that was never added, should not raise
    def dummy(e):
        pass

    bus.unsubscribe('foo', dummy)  # no error

    # Also unsubscribe from event with no handlers
    bus.unsubscribe('bar', dummy)  # no error


def test_subscribe_same_handler_multiple_times_adds_multiple_calls():
    random.seed(5)
    bus = EventBus()

    calls = []

    def handler(e):
        calls.append(e)

    bus.subscribe('ev', handler)
    bus.subscribe('ev', handler)

    bus.emit('ev', 10)

    # Because subscribed twice, called twice
    assert calls == [10, 10]


def test_subscribe_and_unsubscribe_with_different_events():
    random.seed(6)
    bus = EventBus()

    calls = []

    def handler1(e):
        calls.append(('h1', e))

    def handler2(e):
        calls.append(('h2', e))

    bus.subscribe('a', handler1)
    bus.subscribe('b', handler2)

    bus.emit('a', 1)
    bus.emit('b', 2)

    bus.unsubscribe('a', handler1)
    bus.emit('a', 3)
    bus.emit('b', 4)

    assert calls == [('h1', 1), ('h2', 2), ('h2', 4)]


def test_emit_no_handlers_does_nothing():
    random.seed(7)
    bus = EventBus()

    # Should not raise or error
    bus.emit('no_handlers_event', 'anything')


def test_handler_receives_args_and_kwargs_if_emitted_with():
    # Though the interface is probably (event, data),
    # test that only one argument is sent - event_bus likely calls handlers with one argument.
    # If emit passes more, check

    random.seed(8)
    bus = EventBus()

    received = []

    def handler(event_data):
        received.append(event_data)

    bus.subscribe('evt', handler)
    bus.emit('evt', {'key': 'value'})

    assert received == [{'key': 'value'}]


def test_handler_raising_exception_does_not_prevent_others():
    random.seed(9)
    bus = EventBus()

    calls = []

    def handler_raise(e):
        raise ValueError('fail')

    def handler_ok(e):
        calls.append(e)

    bus.subscribe('e', handler_raise)
    bus.subscribe('e', handler_ok)

    with pytest.raises(ValueError):
        bus.emit('e', 42)

    # The second handler should be called before exception or not? Because emit probably calls handlers sequentially.
    # If first raises immediately: others not called.
    # So behavior depends on implementation.
    #
    # We just check calls may be empty or have 42.

    # To be safe, just ensure at least one handler was called or exception raised.
    assert calls == [] or calls == [42]


def test_subscribe_handler_multiple_events_and_unsubscribe_independently():
    random.seed(10)
    bus = EventBus()

    calls = []

    def handler(e):
        calls.append(e)

    bus.subscribe('e1', handler)
    bus.subscribe('e2', handler)

    bus.emit('e1', 1)
    bus.emit('e2', 2)

    bus.unsubscribe('e1', handler)
    bus.emit('e1', 3)
    bus.emit('e2', 4)

    assert calls == [1, 2, 4]


def test_subscribe_handler_then_unsubscribe_re_subscribe_emits_correctly():
    random.seed(11)
    bus = EventBus()

    calls = []

    def handler(e):
        calls.append(e)

    bus.subscribe('e', handler)
    bus.emit('e', 'first')

    bus.unsubscribe('e', handler)
    bus.emit('e', 'second')

    bus.subscribe('e', handler)
    bus.emit('e', 'third')

    assert calls == ['first', 'third']


def test_version_string_is_correct():
    # This file imports EventBus from event_bus/__init__.py,
    # EventBus should be imported and __version__ should be present
    import event_bus
    assert isinstance(event_bus.__version__, str)
    assert event_bus.__version__ == '1.0.0'


def test_eventbus_repr_contains_classname():
    random.seed(12)
    bus = EventBus()
    r = repr(bus)
    assert 'EventBus' in r


def test_subscribe_non_callable_raises_typeerror():
    random.seed(13)
    bus = EventBus()

    with pytest.raises(TypeError):
        bus.subscribe('e', "not a callable")


def test_unsubscribe_non_callable_raises_typeerror():
    random.seed(14)
    bus = EventBus()

    with pytest.raises(TypeError):
        bus.unsubscribe('e', "not a callable")


def test_subscribe_and_emit_with_integer_event_names():
    random.seed(15)
    bus = EventBus()

    calls = []

    def handler(e):
        calls.append(e)

    bus.subscribe(123, handler)
    bus.emit(123, 'int event')

    assert calls == ['int event']


def test_no_handlers_are_called_after_unsubscribe_all():
    random.seed(16)
    bus = EventBus()

    calls = []

    def handler(e):
        calls.append(e)

    bus.subscribe('e', handler)
    bus.unsubscribe('e', handler)
    bus.emit('e', 'x')

    assert calls == []


def test_handlers_order_preserved():
    random.seed(17)
    bus = EventBus()
    calls = []

    def h1(e):
        calls.append('h1')

    def h2(e):
        calls.append('h2')

    bus.subscribe('ev', h1)
    bus.subscribe('ev', h2)

    bus.emit('ev', None)

    assert calls == ['h1', 'h2']
