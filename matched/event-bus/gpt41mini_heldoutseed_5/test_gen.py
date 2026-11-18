import pytest
from event_bus import EventBus


def test_event_bus_can_be_instantiated():
    bus = EventBus()
    assert bus is not None


def test_subscribe_and_publish_single_event():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("test:event", handler)
    bus.publish("test:event", 123)

    assert results == [123]


def test_subscribe_and_publish_multiple_events():
    bus = EventBus()
    results = []

    def handler1(payload):
        results.append(('h1', payload))

    def handler2(payload):
        results.append(('h2', payload))

    bus.subscribe("test:event", handler1)
    bus.subscribe("test:event", handler2)
    bus.publish("test:event", "abc")

    assert ('h1', "abc") in results
    assert ('h2', "abc") in results
    assert len(results) == 2


def test_unsubscribe_removes_handler():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("test:event", handler)
    bus.unsubscribe("test:event", handler)
    bus.publish("test:event", 42)

    assert results == []


def test_unsubscribe_nonexistent_handler_does_not_raise():
    bus = EventBus()

    def handler(payload):
        pass

    # Should not raise even though handler was never subscribed
    bus.unsubscribe("test:event", handler)


def test_publish_returns_true_if_handled():
    bus = EventBus()

    def handler(_):
        pass

    bus.subscribe("test:event", handler)
    result = bus.publish("test:event", None)

    assert result is True


def test_publish_returns_false_if_no_handlers():
    bus = EventBus()
    result = bus.publish("no:event", None)
    assert result is False


def test_subscribe_handler_multiple_times_results_in_multiple_calls():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("test:event", handler)
    bus.subscribe("test:event", handler)
    bus.publish("test:event", "data")

    # Handler called twice because subscribed twice
    assert results == ["data", "data"]


def test_publish_different_event_does_not_call_handler():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("event:a", handler)
    bus.publish("event:b", 99)

    assert results == []


def test_handler_can_subscribe_new_handler_during_publish():
    bus = EventBus()
    results = []

    def new_handler(payload):
        results.append(('new', payload))

    def first_handler(payload):
        results.append(('first', payload))
        bus.subscribe("event", new_handler)

    bus.subscribe("event", first_handler)
    bus.publish("event", 1)
    # new_handler subscribed only after first publish, so only first handler called now

    assert results == [('first', 1)]

    # now new_handler should be called on next publish as well
    results.clear()
    bus.publish("event", 2)

    assert ('new', 2) in results
    assert ('first', 2) in results


def test_handler_can_unsubscribe_itself_during_publish():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)
        bus.unsubscribe("event", handler)

    bus.subscribe("event", handler)
    bus.publish("event", 1)
    # Handler should be called once, then removed

    results.clear()
    bus.publish("event", 2)
    # No calls this time

    assert results == []


def test_subscribe_with_multiple_different_events():
    bus = EventBus()
    results = []

    def handler1(payload):
        results.append(('a', payload))

    def handler2(payload):
        results.append(('b', payload))

    bus.subscribe("event:a", handler1)
    bus.subscribe("event:b", handler2)

    bus.publish("event:a", 1)
    bus.publish("event:b", 2)

    assert ('a', 1) in results
    assert ('b', 2) in results


def test_subscribe_decorator_style():
    bus = EventBus()
    results = []

    @bus.subscribe("event:x")
    def handler(payload):
        results.append(payload)

    bus.publish("event:x", 55)

    assert results == [55]


def test_subscribe_decorator_can_unsubscribe():
    bus = EventBus()
    results = []

    @bus.subscribe("event:x")
    def handler(payload):
        results.append(payload)

    bus.unsubscribe("event:x", handler)
    bus.publish("event:x", 100)

    assert results == []


def test_subscribe_same_function_different_events():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("event:a", handler)
    bus.subscribe("event:b", handler)

    bus.publish("event:a", 1)
    bus.publish("event:b", 2)

    assert 1 in results
    assert 2 in results


def test_unsubscribe_one_of_multiple_handlers():
    bus = EventBus()
    results = []

    def handler1(payload):
        results.append(('h1', payload))

    def handler2(payload):
        results.append(('h2', payload))

    bus.subscribe("event", handler1)
    bus.subscribe("event", handler2)
    bus.unsubscribe("event", handler1)

    bus.publish("event", "data")

    assert ('h1', "data") not in results
    assert ('h2', "data") in results


def test_subscribe_returns_handler_function():
    bus = EventBus()

    def handler(payload):
        pass

    ret = bus.subscribe("event", handler)
    # The subscribe method should return the handler function for decorator compatibility
    assert ret is handler


def test_publish_handler_raises_exception_propagates():
    bus = EventBus()

    def handler(payload):
        raise RuntimeError("fail")

    bus.subscribe("event", handler)

    with pytest.raises(RuntimeError):
        bus.publish("event", "payload")


def test_subscribe_none_handler_raises_type_error():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("event", None)


def test_unsubscribe_none_handler_does_not_raise():
    # Should handle unsubscribing None gracefully even if no handler provided
    bus = EventBus()
    # No error awaited; if error happens test fails
    bus.unsubscribe("event", None)


def test_subscribe_with_empty_event_name():
    bus = EventBus()
    results = []

    def handler(payload):
        results.append(payload)

    bus.subscribe("", handler)
    bus.publish("", 99)

    assert results == [99]
