import pytest
from event_bus import EventBus

def test_eventbus_instance_creation():
    bus = EventBus()
    assert isinstance(bus, EventBus)

def test_subscribe_and_emit():
    bus = EventBus()
    result = []

    def handler(event):
        result.append(event)

    bus.subscribe('test_event', handler)
    bus.emit('test_event', 123)

    assert result == [123]

def test_unsubscribe_handler():
    bus = EventBus()
    result = []

    def handler(event):
        result.append(event)

    bus.subscribe('event', handler)
    bus.unsubscribe('event', handler)
    bus.emit('event', 1)

    assert result == []

def test_subscribe_multiple_handlers():
    bus = EventBus()
    result = []

    def h1(e):
        result.append(('h1', e))

    def h2(e):
        result.append(('h2', e))

    bus.subscribe('ev', h1)
    bus.subscribe('ev', h2)
    bus.emit('ev', 'data')

    assert ('h1', 'data') in result
    assert ('h2', 'data') in result
    assert len(result) == 2

def test_emit_no_subscribers_does_not_error():
    bus = EventBus()
    # Should not raise even if no handlers subscribed
    bus.emit('no_subscribers', 42)

def test_unsubscribe_nonexistent_handler_does_not_error():
    bus = EventBus()

    def handler(e):
        pass

    # Unsubscribe handler not subscribed, no error
    bus.unsubscribe('some_event', handler)

def test_subscribe_same_handler_multiple_times_only_called_once():
    bus = EventBus()
    calls = 0

    def handler(e):
        nonlocal calls
        calls += 1

    bus.subscribe('event', handler)
    bus.subscribe('event', handler)
    bus.emit('event', None)

    # Depending on implementation, duplicates may be allowed or ignored.
    # Assuming duplicates are ignored and handler called once.
    assert calls == 1

def test_emit_passes_correct_event_data():
    bus = EventBus()

    received = []

    def handler(data):
        received.append(data)

    bus.subscribe('my_event', handler)
    bus.emit('my_event', {'key': 'value'})

    assert received == [{'key': 'value'}]

def test_subscribe_different_events():
    bus = EventBus()
    collected = {}

    def h1(e):
        collected['a'] = e

    def h2(e):
        collected['b'] = e

    bus.subscribe('event_a', h1)
    bus.subscribe('event_b', h2)

    bus.emit('event_a', 1)
    bus.emit('event_b', 2)

    assert collected == {'a': 1, 'b': 2}

def test_handlers_are_independent_between_events():
    bus = EventBus()
    called = []

    def h1(e):
        called.append('h1')

    def h2(e):
        called.append('h2')

    bus.subscribe('event1', h1)
    bus.subscribe('event2', h2)

    bus.emit('event1', None)
    assert called == ['h1']

    called.clear()
    bus.emit('event2', None)
    assert called == ['h2']

def test_unsubscribe_one_of_multiple_handlers_only_removes_that_one():
    bus = EventBus()
    called = []

    def h1(e):
        called.append('h1')

    def h2(e):
        called.append('h2')

    bus.subscribe('ev', h1)
    bus.subscribe('ev', h2)

    bus.unsubscribe('ev', h1)
    bus.emit('ev', None)

    assert called == ['h2']

def test_subscribe_none_handler_raises():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe('event', None)

def test_unsubscribe_none_handler_raises():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.unsubscribe('event', None)

def test_subscribe_handler_called_in_order():
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

def test_emit_with_no_event_name():
    bus = EventBus()
    # Passing None or empty string as event name might raise or be allowed
    # Test with None
    with pytest.raises(TypeError):
        bus.subscribe(None, lambda e: None)

    with pytest.raises(TypeError):
        bus.emit(None, 1)

def test_multiple_events_and_handlers_mix():
    bus = EventBus()
    called = []

    def ha(e):
        called.append(f'ha:{e}')

    def hb(e):
        called.append(f'hb:{e}')

    bus.subscribe('a', ha)
    bus.subscribe('b', hb)

    bus.emit('a', 1)
    bus.emit('b', 2)
    bus.emit('a', 3)

    assert called == ['ha:1', 'hb:2', 'ha:3']

def test_subscribe_handler_returns_none():
    bus = EventBus()
    def handler(e):
        pass
    r = bus.subscribe('event', handler)
    assert r is None

def test_unsubscribe_handler_returns_none():
    bus = EventBus()
    def handler(e):
        pass
    bus.subscribe('event', handler)
    r = bus.unsubscribe('event', handler)
    assert r is None

def test_emit_handler_raises_exception_does_not_stop_other_handlers():
    bus = EventBus()
    called = []

    def h1(e):
        raise ValueError("fail")

    def h2(e):
        called.append('h2')

    bus.subscribe('ev', h1)
    bus.subscribe('ev', h2)

    # The best practice for event bus is handlers are called independently,
    # even if one raises, others will still be called.
    # But depending on implementation this might not be true.
    # We test that no exception escapes emit.
    try:
        bus.emit('ev', 123)
    except Exception:
        pytest.fail("emit should not raise even if handler raises")

    assert called == ['h2']

def test_subscribe_with_non_string_event_name_raises():
    bus = EventBus()
    def handler(e):
        pass
    with pytest.raises(TypeError):
        bus.subscribe(123, handler)

def test_unsubscribe_with_non_string_event_name_raises():
    bus = EventBus()
    def handler(e):
        pass
    with pytest.raises(TypeError):
        bus.unsubscribe(123, handler)
