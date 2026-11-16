"""Unit tests for event_bus.EventBus.

These tests exercise the public EventBus API using the method names
subscribe(event_name, callback), publish(event_name, *args, **kwargs),
and unsubscribe(event_name, callback). Only pytest and the standard
library are used.
"""
import pytest
from unittest.mock import Mock

import event_bus
from event_bus import EventBus


@pytest.fixture
def bus():
    return EventBus()


def test_package_exports_eventbus_class():
    assert "EventBus" in event_bus.__all__
    assert event_bus.EventBus is EventBus


def test_package_version_is_defined():
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_can_be_instantiated():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_publish_invokes_subscribed_callback(bus):
    callback = Mock()
    bus.subscribe("user.created", callback)

    bus.publish("user.created")

    callback.assert_called_once()


def test_publish_passes_positional_arguments(bus):
    callback = Mock()
    bus.subscribe("order.placed", callback)

    bus.publish("order.placed", 42, "widget")

    callback.assert_called_once_with(42, "widget")


def test_publish_passes_keyword_arguments(bus):
    callback = Mock()
    bus.subscribe("order.placed", callback)

    bus.publish("order.placed", quantity=3, sku="ABC-1")

    callback.assert_called_once_with(quantity=3, sku="ABC-1")


def test_publish_notifies_all_subscribers(bus):
    first = Mock()
    second = Mock()
    third = Mock()
    bus.subscribe("tick", first)
    bus.subscribe("tick", second)
    bus.subscribe("tick", third)

    bus.publish("tick")

    first.assert_called_once()
    second.assert_called_once()
    third.assert_called_once()


def test_subscribers_are_invoked_in_registration_order(bus):
    calls = []
    bus.subscribe("step", lambda: calls.append("a"))
    bus.subscribe("step", lambda: calls.append("b"))
    bus.subscribe("step", lambda: calls.append("c"))

    bus.publish("step")

    assert calls == ["a", "b", "c"]


def test_publish_only_reaches_subscribers_of_that_event(bus):
    created = Mock()
    deleted = Mock()
    bus.subscribe("item.created", created)
    bus.subscribe("item.deleted", deleted)

    bus.publish("item.created")

    created.assert_called_once()
    deleted.assert_not_called()


def test_publish_without_subscribers_does_not_raise(bus):
    bus.publish("nobody.listening", 1, 2, key="value")


def test_publish_multiple_times_invokes_callback_each_time(bus):
    callback = Mock()
    bus.subscribe("heartbeat", callback)

    for _ in range(5):
        bus.publish("heartbeat")

    assert callback.call_count == 5


def test_each_publish_delivers_its_own_arguments(bus):
    callback = Mock()
    bus.subscribe("value", callback)

    bus.publish("value", 1)
    bus.publish("value", 2)

    assert callback.call_args_list == [((1,), {}), ((2,), {})]


def test_unsubscribe_stops_delivery(bus):
    callback = Mock()
    bus.subscribe("alert", callback)
    bus.unsubscribe("alert", callback)

    bus.publish("alert")

    callback.assert_not_called()


def test_unsubscribe_only_removes_given_callback(bus):
    kept = Mock()
    removed = Mock()
    bus.subscribe("alert", kept)
    bus.subscribe("alert", removed)

    bus.unsubscribe("alert", removed)
    bus.publish("alert")

    kept.assert_called_once()
    removed.assert_not_called()


def test_unsubscribe_only_affects_the_named_event(bus):
    callback = Mock()
    bus.subscribe("event.a", callback)
    bus.subscribe("event.b", callback)

    bus.unsubscribe("event.a", callback)
    bus.publish("event.a")
    bus.publish("event.b")

    callback.assert_called_once()


def test_separate_bus_instances_are_isolated():
    bus_one = EventBus()
    bus_two = EventBus()
    callback = Mock()
    bus_one.subscribe("shared.name", callback)

    bus_two.publish("shared.name")

    callback.assert_not_called()


def test_callback_exception_propagates_to_publisher(bus):
    def failing(*_args, **_kwargs):
        raise RuntimeError("boom")

    bus.subscribe("explode", failing)

    with pytest.raises(RuntimeError, match="boom"):
        bus.publish("explode")
