import pytest
from event_bus import EventBus


def test_eventbus_can_be_instantiated():
    bus = EventBus()
    assert isinstance(bus, EventBus)


def test_subscribe_and_emit_single_event():
    bus = EventBus()
    events = []

    def handler(data):
        events.append(data)

    bus.subscribe("test_event", handler)
    bus.emit("test_event", 123)

    assert events == [123]


def test_emit_without_subscribers():
    bus = EventBus()
    # Should not raise any exception
    bus.emit("unsubscribed_event", "data")


def test_subscribe_multiple_handlers_same_event():
    bus = EventBus()
    results = []

    def handler1(data):
        results.append(("h1", data))

    def handler2(data):
        results.append(("h2", data))

    bus.subscribe("event", handler1)
    bus.subscribe("event", handler2)
    bus.emit("event", "hello")

    assert ("h1", "hello") in results
    assert ("h2", "hello") in results
    assert len(results) == 2


def test_subscribe_same_handler_multiple_times():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe("ev", handler)
    bus.subscribe("ev", handler)  # subscribing twice
    bus.emit("ev", 1)

    # The handler should be called twice if subscribed twice
    assert calls == [1, 1]


def test_unsubscribe_removes_handler():
    bus = EventBus()
    results = []

    def handler(data):
        results.append(data)

    bus.subscribe("e", handler)
    bus.unsubscribe("e", handler)
    bus.emit("e", "no_call")

    assert results == []


def test_unsubscribe_only_one_handler_when_multiple_subscribed():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe("e", handler)
    bus.subscribe("e", handler)
    bus.unsubscribe("e", handler)
    bus.emit("e", 42)

    # One subscription removed, one still remains
    assert calls == [42]


def test_unsubscribe_nonexistent_handler_no_effect():
    bus = EventBus()
    # Should not raise
    def handler(data):
        pass

    bus.unsubscribe("nonexistent_event", handler)
    bus.unsubscribe("event", handler)


def test_subscribe_handler_receives_all_emit_args_and_kwargs():
    bus = EventBus()
    results = []

    def handler(a, b, c=None):
        results.append((a, b, c))

    bus.subscribe("args_event", handler)
    bus.emit("args_event", 1, 2, c=3)

    assert results == [(1, 2, 3)]


def test_emit_multiple_events_calls_correct_handlers():
    bus = EventBus()
    calls = []

    def handler_a(data):
        calls.append(("a", data))

    def handler_b(data):
        calls.append(("b", data))

    bus.subscribe("event_a", handler_a)
    bus.subscribe("event_b", handler_b)

    bus.emit("event_a", "foo")
    bus.emit("event_b", "bar")

    assert ("a", "foo") in calls
    assert ("b", "bar") in calls
    assert len(calls) == 2


def test_subscribe_unsubscribe_and_re_subscribe():
    bus = EventBus()
    calls = []

    def h(data):
        calls.append(data)

    bus.subscribe("e", h)
    bus.emit("e", 1)
    bus.unsubscribe("e", h)
    bus.emit("e", 2)
    bus.subscribe("e", h)
    bus.emit("e", 3)

    assert calls == [1, 3]


def test_handler_raising_exception_does_not_stop_other_handlers():
    bus = EventBus()
    calls = []

    def bad_handler(data):
        raise RuntimeError("fail")

    def good_handler(data):
        calls.append(data)

    bus.subscribe("e", bad_handler)
    bus.subscribe("e", good_handler)

    # The exception in bad_handler should propagate, or should it be caught?
    # Not specified in source provided, so we test if exception propagates
    with pytest.raises(RuntimeError, match="fail"):
        bus.emit("e", 123)

    # good_handler should not be called because bad_handler raised before
    assert calls == []


def test_handler_raising_exception_with_multiple_handlers_order():
    bus = EventBus()
    calls = []

    def handler1(data):
        calls.append(1)

    def handler2(data):
        raise ValueError("fail")

    def handler3(data):
        calls.append(3)

    bus.subscribe("event", handler1)
    bus.subscribe("event", handler2)
    bus.subscribe("event", handler3)

    with pytest.raises(ValueError, match="fail"):
        bus.emit("event", None)

    # handler1 called, handler2 raised, handler3 not called
    assert calls == [1]


def test_unsubscribing_handler_multiple_times_no_error():
    bus = EventBus()

    def handler(data):
        pass

    bus.subscribe("e", handler)
    bus.unsubscribe("e", handler)
    # second unsubscribe shouldn't cause error
    bus.unsubscribe("e", handler)


def test_subscribe_handler_for_multiple_events():
    bus = EventBus()
    calls = []

    def handler(data):
        calls.append(data)

    bus.subscribe("event1", handler)
    bus.subscribe("event2", handler)

    bus.emit("event1", "a")
    bus.emit("event2", "b")

    assert calls == ["a", "b"]


def test_handlers_are_independent_between_events():
    bus = EventBus()
    calls_e1 = []
    calls_e2 = []

    def h1(d):
        calls_e1.append(d)

    def h2(d):
        calls_e2.append(d)

    bus.subscribe("e1", h1)
    bus.subscribe("e2", h2)

    bus.emit("e1", 1)
    bus.emit("e2", 2)

    assert calls_e1 == [1]
    assert calls_e2 == [2]
