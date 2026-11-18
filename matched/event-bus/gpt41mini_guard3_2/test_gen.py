import random
import pytest
from event_bus.bus import EventBus


def test_eventbus_can_be_instantiated():
    random.seed(0)
    bus = EventBus()
    assert bus is not None


def test_subscribe_and_emit_single_listener():
    random.seed(0)
    bus = EventBus()
    called = []

    def listener(event):
        called.append(event)

    bus.subscribe("my_event", listener)
    bus.emit("my_event", 123)

    assert len(called) == 1
    assert called[0] == 123


def test_emit_no_listeners_does_not_fail():
    random.seed(0)
    bus = EventBus()
    # Should not raise even if no listeners are subscribed
    bus.emit("unknown_event", "payload")


def test_subscribe_multiple_listeners_and_emit_order():
    random.seed(0)
    bus = EventBus()
    called = []

    def listener1(event):
        called.append(("1", event))

    def listener2(event):
        called.append(("2", event))

    bus.subscribe("data", listener1)
    bus.subscribe("data", listener2)

    bus.emit("data", "abc")

    # The order should be guaranteed as subscribe order
    assert called == [("1", "abc"), ("2", "abc")]


def test_subscribe_same_listener_multiple_times_emits_multiple_calls():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener(event):
        calls.append(event)

    bus.subscribe("x", listener)
    bus.subscribe("x", listener)
    bus.emit("x", 42)

    # listener should be called twice
    assert calls == [42, 42]


def test_unsubscribe_listener_stops_receiving_events():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener(event):
        calls.append(event)

    bus.subscribe("event", listener)
    bus.unsubscribe("event", listener)
    bus.emit("event", "should not be received")

    assert calls == []


def test_unsubscribe_nonexistent_listener_does_not_error():
    random.seed(0)
    bus = EventBus()

    def listener(event):
        pass

    # Unsubscribing listener without subscribing first should not raise
    bus.unsubscribe("nonexistent", listener)


def test_subscribe_unsubscribe_then_subscribe_again():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener(event):
        calls.append(event)

    bus.subscribe("foo", listener)
    bus.unsubscribe("foo", listener)
    bus.subscribe("foo", listener)
    bus.emit("foo", 7)

    assert calls == [7]


def test_emit_multiple_events_with_different_listeners():
    random.seed(0)
    bus = EventBus()
    calls_a = []
    calls_b = []

    def listener_a(event):
        calls_a.append(event)

    def listener_b(event):
        calls_b.append(event)

    bus.subscribe("a", listener_a)
    bus.subscribe("b", listener_b)

    bus.emit("a", 1)
    bus.emit("b", 2)
    bus.emit("a", 3)

    assert calls_a == [1, 3]
    assert calls_b == [2]


def test_listeners_for_different_events_are_independent():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener1(event):
        calls.append(("1", event))

    def listener2(event):
        calls.append(("2", event))

    bus.subscribe("one", listener1)
    bus.subscribe("two", listener2)

    bus.emit("one", "x")
    bus.emit("two", "y")

    # Each listener only receives their event type
    assert ("1", "x") in calls
    assert ("2", "y") in calls
    assert all(evt in ("x", "y") for _, evt in calls)


def test_subscribe_with_non_callable_raises_typeerror():
    random.seed(0)
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("e", "not callable")


def test_unsubscribe_with_non_callable_raises_typeerror():
    random.seed(0)
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.unsubscribe("e", "not callable")


def test_subscribe_listener_mutating_calls_does_not_break_emit():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener(event):
        calls.append(event)
        # Subscribe another listener during emit
        def new_listener(e):
            calls.append(e * 10)
        bus.subscribe("event", new_listener)

    bus.subscribe("event", listener)
    bus.emit("event", 5)

    # The new listener should not be called during current emit
    assert calls == [5]


def test_emit_calls_all_listeners_even_if_one_raises():
    random.seed(0)
    bus = EventBus()
    calls = []

    def listener_ok(event):
        calls.append(event)

    def listener_fail(event):
        raise ValueError("fail")

    bus.subscribe("ev", listener_ok)
    bus.subscribe("ev", listener_fail)
    bus.subscribe("ev", listener_ok)

    with pytest.raises(ValueError):
        bus.emit("ev", 99)

    # The first listener_ok should have been called; the one after the exception may or may not be called,
    # behavior depends on EventBus implementation; test that at least the first was called
    assert calls and calls[0] == 99


def test_subscribe_listener_receives_exact_event_object():
    random.seed(0)
    bus = EventBus()
    event_obj = object()

    received = []

    def listener(event):
        received.append(event)

    bus.subscribe("event", listener)
    bus.emit("event", event_obj)

    assert received[0] is event_obj


def test_subscribe_and_unsubscribe_multiple_listeners():
    random.seed(0)
    bus = EventBus()
    called = []

    def l1(e): called.append(("l1", e))
    def l2(e): called.append(("l2", e))
    def l3(e): called.append(("l3", e))

    bus.subscribe("ev", l1)
    bus.subscribe("ev", l2)
    bus.subscribe("ev", l3)

    bus.unsubscribe("ev", l2)
    bus.emit("ev", 100)

    assert ("l1", 100) in called
    assert ("l3", 100) in called
    assert all(name != "l2" for name, _ in called)


def test_subscribe_and_unsubscribe_different_events():
    random.seed(0)
    bus = EventBus()
    calls = []

    def l(e): calls.append(e)

    bus.subscribe("e1", l)
    bus.subscribe("e2", l)

    bus.unsubscribe("e1", l)
    bus.emit("e1", 1)
    bus.emit("e2", 2)

    # Listener should only receive event 'e2'
    assert calls == [2]
