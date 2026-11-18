import pytest
from event_bus.bus import EventBus


def test_eventbus_init_creates_empty_listeners_dict():
    bus = EventBus()
    assert isinstance(bus._listeners, dict)
    assert len(bus._listeners) == 0


def test_subscribe_adds_listener():
    bus = EventBus()
    results = []

    def listener(event):
        results.append(event)

    bus.subscribe("my_event", listener)
    assert "my_event" in bus._listeners
    assert listener in bus._listeners["my_event"]


def test_subscribe_multiple_listeners_for_same_event():
    bus = EventBus()
    events_received = []

    def l1(e): events_received.append(("l1", e))
    def l2(e): events_received.append(("l2", e))

    bus.subscribe("e", l1)
    bus.subscribe("e", l2)
    assert l1 in bus._listeners["e"]
    assert l2 in bus._listeners["e"]


def test_subscribe_raises_if_listener_not_callable():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("event", "not_callable")


def test_unsubscribe_removes_listener():
    bus = EventBus()
    called = []

    def listener(e):
        called.append(e)

    bus.subscribe("evt", listener)
    bus.unsubscribe("evt", listener)
    assert listener not in bus._listeners.get("evt", [])


def test_unsubscribe_nonexistent_listener_does_not_raise():
    bus = EventBus()

    def listener(e):
        pass

    # Unsubscribing listener that was never added should just do nothing
    bus.unsubscribe("evt", listener)
    # Also if event was never subscribed to
    bus.unsubscribe("nonexistent_evt", listener)


def test_publish_calls_listeners_in_order():
    bus = EventBus()
    calls = []

    def l1(e):
        calls.append("l1")

    def l2(e):
        calls.append("l2")

    bus.subscribe("event", l1)
    bus.subscribe("event", l2)

    bus.publish("event", 42)
    assert calls == ["l1", "l2"]


def test_publish_passes_event_argument():
    bus = EventBus()
    received = []

    def listener(e):
        received.append(e)

    bus.subscribe("evt", listener)
    bus.publish("evt", "payload")
    assert received == ["payload"]


def test_publish_event_with_no_listeners_does_nothing():
    bus = EventBus()
    # Should not raise
    bus.publish("no_listener_event", "data")


def test_subscribe_same_listener_multiple_times_adds_multiple_calls():
    bus = EventBus()
    calls = []

    def listener(e):
        calls.append(e)

    bus.subscribe("e", listener)
    bus.subscribe("e", listener)

    bus.publish("e", 10)

    # listener should be called twice
    assert calls == [10, 10]


def test_unsubscribe_removes_only_one_occurrence_of_listener():
    bus = EventBus()
    calls = []

    def listener(e):
        calls.append(e)

    bus.subscribe("e", listener)
    bus.subscribe("e", listener)

    bus.unsubscribe("e", listener)

    bus.publish("e", 1)
    # Only one listener call left
    assert calls == [1]


def test_unsubscribe_all_listeners_for_event_removes_key_when_empty():
    bus = EventBus()

    def listener(e):
        pass

    bus.subscribe("event", listener)
    bus.unsubscribe("event", listener)

    # After removal, listeners list should be empty or key not present
    assert "event" not in bus._listeners or len(bus._listeners["event"]) == 0


def test_listeners_can_subscribe_other_listeners_during_publish():
    bus = EventBus()
    calls = []

    def listener1(event):
        calls.append("l1")
        # Subscribe listener2 during publish
        bus.subscribe("test", listener2)

    def listener2(event):
        calls.append("l2")

    bus.subscribe("test", listener1)
    bus.publish("test", None)
    # listener2 subscribed too late to be called this time
    assert calls == ["l1"]

    bus.publish("test", None)
    assert calls == ["l1", "l1", "l2"]


def test_listener_unsubscribes_itself_during_publish():
    bus = EventBus()
    calls = []

    def listener(event):
        calls.append("called")
        bus.unsubscribe("ev", listener)

    bus.subscribe("ev", listener)
    bus.publish("ev", None)
    # Listener called once
    assert calls == ["called"]

    calls.clear()
    bus.publish("ev", None)
    # Listener unsubscribed, so not called again
    assert calls == []


def test_subscribe_with_empty_event_name_allows_subscription():
    bus = EventBus()
    calls = []

    def listener(e):
        calls.append(e)

    bus.subscribe("", listener)
    bus.publish("", "data")
    assert calls == ["data"]


def test_subscribe_with_none_event_name_raises():
    bus = EventBus()
    def listener(e):
        pass
    with pytest.raises(TypeError):
        bus.subscribe(None, listener)


def test_unsubscribe_with_none_event_name_raises():
    bus = EventBus()
    def listener(e):
        pass
    with pytest.raises(TypeError):
        bus.unsubscribe(None, listener)


def test_publish_with_none_event_name_raises():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.publish(None, "data")


def test_unsubscribe_listener_not_callable_raises():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.unsubscribe("evt", "not_callable")


def test_listeners_called_with_event_data_of_various_types():
    bus = EventBus()
    results = []

    def listener(e):
        results.append(e)

    bus.subscribe("evt", listener)

    bus.publish("evt", 123)
    bus.publish("evt", {"key": "value"})
    bus.publish("evt", [1, 2, 3])
    bus.publish("evt", None)

    assert results == [123, {"key": "value"}, [1, 2, 3], None]
